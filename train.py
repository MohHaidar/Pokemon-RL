"""
train.py – Train a PPO agent on Pokemon Red.

Uses RecurrentPPO with MultiInputLstmPolicy so the agent has memory across
steps and sees both raw pixels and a structured game-state vector.

Usage
-----
Start fresh:
    python train.py

Resume from checkpoint:
    python train.py --resume runs/checkpoints/pokemon_ppo_500000_steps

Change envs / total steps:
    python train.py --envs 2 --steps 5_000_000

Monitor:
    tensorboard --logdir logs

NOTE: Observation space changed from plain CnnPolicy — old checkpoints are
NOT compatible. Always start fresh after this upgrade.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import gymnasium as gym
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from env import PokemonRedEnv


class PokemonCnnExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the Pokemon Red environment.

    Processes the three observation keys separately then concatenates:
      - "screen"  (84×84×4 grayscale frame-stack)  → CNN        → 256-dim
      - "state"   (91-dim float vector)             → MLP(2-layer) → 64-dim
      - "minimap" (21×21×1 visited-tile grid)       → CNN        → 64-dim
      Concatenated output: 384-dim → 2-layer LSTM(256)
    """

    CNN_OUT     = 256
    STATE_OUT   = 64
    MINIMAP_OUT = 64

    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=self.CNN_OUT + self.STATE_OUT + self.MINIMAP_OUT)

        screen_space = observation_space["screen"]
        # SB3 wraps SubprocVecEnv with VecTransposeImage → space is already CHW (C, H, W)
        n_channels = screen_space.shape[0]  # 4 stacked frames

        # Small but expressive CNN — 3 conv layers with batch-norm
        self.cnn = nn.Sequential(
            # 84×84 → 20×20
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # 20×20 → 9×9
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # 9×9 → 7×7
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),                            # 64 × 7 × 7 = 3136
            nn.Linear(3136, self.CNN_OUT),
            nn.ReLU(),
        )

        state_dim = observation_space["state"].shape[0]
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.STATE_OUT),
            nn.ReLU(),
        )

        # Small CNN for the 21×21 visited-tile minimap (1 channel after VecTransposeImage)
        # (1, 21, 21) → Conv s2 → (8, 10, 10) → Conv s2 → (16, 4, 4) → Linear → 64
        self.minimap_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=0),   # → (8, 10, 10)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0),  # → (16, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                                           # 16 × 4 × 4 = 256
            nn.Linear(256, self.MINIMAP_OUT),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # SB3 already converts to CHW float [0,1] via VecTransposeImage + preprocess_obs
        cnn_out     = self.cnn(observations["screen"])
        state_out   = self.state_net(observations["state"])
        minimap_out = self.minimap_cnn(observations["minimap"])
        return torch.cat([cnn_out, state_out, minimap_out], dim=1)

RUNS_DIR       = Path("runs")
LOGS_DIR       = Path("logs")
CHECKPOINT_DIR = RUNS_DIR / "checkpoints"
BEST_DIR       = RUNS_DIR / "best"


class ScoreThresholdCallback(BaseCallback):
    """Save a separate 'best' checkpoint only when ep_rew_mean exceeds the threshold."""

    def __init__(self, threshold: float, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.threshold  = threshold
        self.save_path  = save_path
        self._best_mean = -float("inf")

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) == 0:
            return True
        mean_reward = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
        if mean_reward >= self.threshold and mean_reward > self._best_mean:
            self._best_mean = mean_reward
            self.model.save(self.save_path)
            if self.verbose:
                print(f"\n[best] New best mean reward {mean_reward:.2f} — saved to {self.save_path}")
        return True


class RewardBreakdownCallback(BaseCallback):
    """Log per-episode reward breakdown keys to tensorboard.

    Accumulates reward_breakdown dicts from info across all envs, then logs
    the per-episode mean of every key once per rollout (every n_steps * n_envs
    environment steps).  Keys appear in tensorboard under rewards/<key>.
    """

    def __init__(self):
        super().__init__(verbose=0)
        # ep_totals[env_idx][key] = running sum for current episode
        self._ep_totals: list[dict[str, float]] = []
        # completed episode totals waiting to be averaged
        self._finished: list[dict[str, float]] = []

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_totals = [{} for _ in range(n)]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))
        for i, (info, done) in enumerate(zip(infos, dones)):
            bd = info.get("reward_breakdown", {})
            for k, v in bd.items():
                self._ep_totals[i][k] = self._ep_totals[i].get(k, 0.0) + v
            if done:
                self._finished.append(dict(self._ep_totals[i]))
                self._ep_totals[i] = {}
        return True

    def _on_rollout_end(self) -> None:
        if not self._finished:
            return
        # Aggregate: mean across completed episodes
        all_keys: set[str] = set()
        for ep in self._finished:
            all_keys.update(ep.keys())
        for key in all_keys:
            vals = [ep[key] for ep in self._finished if key in ep]
            if vals:
                self.logger.record(f"rewards/{key}", sum(vals) / len(vals))
        self._finished.clear()


def make_env(rank: int, seed: int = 0, rom_path: str = "Pokemon_Red.gb", max_steps: int = 8_192):
    def _init():
        env = PokemonRedEnv(rom_path=rom_path, headless=True, max_steps=max_steps)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    rom_path: str        = "Pokemon_Red.gb",
    n_envs: int          = 12,
    total_timesteps: int = 10_000_000,
    resume: str | None   = None,
    seed: int            = 42,
    score_threshold: float = 0.0,   # only save 'best' checkpoints above this score
    max_steps: int       = 8_192,   # episode length — increase gradually as agent matures
) -> None:
    for d in (RUNS_DIR, LOGS_DIR, CHECKPOINT_DIR, BEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[train] Spawning {n_envs} parallel environment(s)...")
    vec_env = SubprocVecEnv([make_env(i, seed, rom_path, max_steps) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    batch_size_info = max(256, n_envs * 128)
    print(f"[train] Environments ready. batch_size={batch_size_info}, max_steps={max_steps}")

    if resume:
        resume_path = Path(resume)
        if not resume_path.exists() and not Path(resume + ".zip").exists():
            raise FileNotFoundError(
                f"Resume file not found: {resume} (tried with and without .zip)\n"
                f"Check the path and try again."
            )
        print(f"[train] Resuming from {resume} ...")
        # Scale batch_size with n_envs to keep ~48 minibatches per epoch
        # buffer_size = n_steps × n_envs = 2048 × n_envs
        # batch_size should divide evenly: 2048 × n_envs / batch_size ≈ 48
        batch_size = max(256, (n_envs * 2048) // 48)
        # Round to clean power-of-2-ish values
        if batch_size > 256:
            batch_size = 2 ** round(np.log2(batch_size))
        print(f"[train] Adjusted batch_size={batch_size} for {n_envs} envs (buffer={n_envs*2048})")
        
        model = RecurrentPPO.load(
            resume,
            env=vec_env,
            custom_objects={
                "batch_size":    batch_size,
                "ent_coef":      0.02,
                "learning_rate": 1.0e-4,
                "n_steps":       2048,
                "n_epochs":      8,
                "clip_range":    0.15,
                "target_kl":     0.02,
                "verbose":       1,
            },
        )
        model.verbose       = 1
        model.tensorboard_log = str(LOGS_DIR)
    else:
        batch_size = 256  # fixed: cleanly divides n_steps*n_envs for typical configs
        model = RecurrentPPO(
            policy          = "MultiInputLstmPolicy",
            env             = vec_env,
            n_steps         = 2048,      # 2048*6=12288 buffer; 12288/256=48 minibatches
            batch_size      = batch_size,
            n_epochs        = 8,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.15,
            ent_coef        = 0.02,
            learning_rate   = 1.0e-4,
            target_kl       = 0.05,    # generous on fresh start — random weights → high variance advantages
            verbose         = 1,
            tensorboard_log = str(LOGS_DIR),
            seed            = seed,
            policy_kwargs   = dict(
                features_extractor_class  = PokemonCnnExtractor,
                features_extractor_kwargs = {},
                lstm_hidden_size          = 256,
                n_lstm_layers             = 2,
                shared_lstm               = False,
            ),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(50_000 // n_envs, 1),
        save_path   = str(CHECKPOINT_DIR),
        name_prefix = "pokemon_ppo",
        verbose     = 1,
    )
    best_cb = ScoreThresholdCallback(
        threshold = score_threshold,
        save_path = str(BEST_DIR / "pokemon_ppo_best"),
        verbose   = 1,
    )
    breakdown_cb = RewardBreakdownCallback()

    print(f"[train] Training for {total_timesteps:,} timesteps...")
    print("[train] Press Ctrl+C at any time to save and exit cleanly.")
    try:
        model.learn(
            total_timesteps     = total_timesteps,
            callback            = [checkpoint_cb, best_cb, breakdown_cb],
            reset_num_timesteps = True,
            tb_log_name         = "PPO",
        )
    except KeyboardInterrupt:
        print("\n[train] Interrupted — saving current model...")

    final_path = RUNS_DIR / "pokemon_ppo_final"
    model.save(str(final_path))
    print(f"[train] Model saved → {final_path}.zip")
    try:
        vec_env.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Pokemon Red PPO agent")
    parser.add_argument("--rom",       default="Pokemon_Red.gb")
    parser.add_argument("--envs",      type=int, default=12)
    parser.add_argument("--steps",     type=int, default=10_000_000)
    parser.add_argument("--resume",    default=None)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Only save 'best' checkpoint when mean reward exceeds this (default: 0)")
    parser.add_argument("--max-steps", type=int, default=8_192,
                        help="Episode length cap. Start small (e.g. 1024) for early training, "
                             "increase on resume as the agent matures (default: 8192)")
    args = parser.parse_args()

    train(
        rom_path        = args.rom,
        n_envs          = args.envs,
        total_timesteps = args.steps,
        resume          = args.resume,
        seed            = args.seed,
        score_threshold = args.threshold,
        max_steps       = args.max_steps,
    )
