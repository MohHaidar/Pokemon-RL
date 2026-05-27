#!/usr/bin/env python3
"""
bc_pretrain.py — Behavioral cloning pre-trainer from human recordings.

Loads .pkl files saved by record.py, trains the exact same RecurrentPPO
policy as train.py via cross-entropy (NLL) on your actions, then saves
a model file that train.py --resume can load for PPO fine-tuning.

The LSTM is present in the architecture but treated as stateless during BC
(every step is an independent episode start with zero hidden state).  The
LSTM weights will develop meaningful memory once PPO fine-tuning starts.

Usage:
    uv run python bc_pretrain.py                         # default settings
    uv run python bc_pretrain.py --epochs 30             # more epochs
    uv run python bc_pretrain.py --recordings demos/     # custom dir
    uv run python bc_pretrain.py --no-noop               # skip no-op steps

After training, run PPO on top of the pre-trained weights:
    uv run python train.py --resume runs/bc_pretrained
"""

from __future__ import annotations

import argparse
import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from train import PokemonCnnExtractor
from env import PokemonRedEnv, ACTIONS

# ── Action names for reporting ─────────────────────────────────────────────────
_ACTION_NAMES = ["noop"] + ACTIONS   # index 0 = noop


# ── Dataset ───────────────────────────────────────────────────────────────────

class DemoDataset(Dataset):
    """Flat (obs_dict, action) pairs loaded from all recording .pkl files."""

    def __init__(self, recordings_dir: str, skip_noop: bool = False):
        paths = sorted(glob.glob(f"{recordings_dir}/*.pkl"))
        if not paths:
            raise FileNotFoundError(
                f"No .pkl recording files found in '{recordings_dir}/'\n"
                f"Record some gameplay first:  uv run python record.py"
            )

        screens, states, minimaps, acts_list = [], [], [], []
        total_steps = 0

        for path in paths:
            with open(path, "rb") as f:
                data = pickle.load(f)
            obs  = data["observations"]   # dict of arrays
            acts = data["actions"]        # (N,) int64
            if skip_noop:
                mask = acts != 0
                obs  = {k: v[mask] for k, v in obs.items()}
                acts = acts[mask]
            screens.append(obs["screen"])
            states.append(obs["state"])
            minimaps.append(obs["minimap"])
            acts_list.append(acts)
            total_steps += len(acts)

        self.screen  = np.concatenate(screens)    # (N, 84, 84, 4)  uint8
        self.state   = np.concatenate(states)     # (N, 57)         float32
        self.minimap = np.concatenate(minimaps)   # (N, 21, 21, 1)  uint8
        self.acts    = np.concatenate(acts_list)  # (N,)            int64

        print(f"  Loaded {len(paths)} recording(s): {total_steps:,} raw steps"
              + (f", {len(self.acts):,} after noop filter" if skip_noop else ""))
        print("  Action distribution:")
        for i, name in enumerate(_ACTION_NAMES):
            n = int((self.acts == i).sum())
            bar = "█" * int(50 * n / max(total_steps, 1))
            print(f"    {i} {name:7s}  {n:6d}  {100*n/max(total_steps,1):5.1f}%  {bar}")

    def __len__(self) -> int:
        return len(self.acts)

    def __getitem__(self, idx):
        # HWC → CHW (SB3/VecTransposeImage convention) + normalise to [0, 1]
        screen  = torch.from_numpy(
            self.screen[idx].transpose(2, 0, 1).astype(np.float32) / 255.0)
        state   = torch.from_numpy(self.state[idx])
        minimap = torch.from_numpy(
            self.minimap[idx].transpose(2, 0, 1).astype(np.float32) / 255.0)
        act = torch.tensor(int(self.acts[idx]), dtype=torch.long)
        return screen, state, minimap, act


# ── Training ──────────────────────────────────────────────────────────────────

def bc_train(
    recordings_dir: str = "recordings",
    output_path:    str = "runs/bc_pretrained",
    resume:         str | None = None,
    n_epochs:       int = 15,
    batch_size:     int = 256,
    lr:           float = 3e-4,
    skip_noop:     bool = False,
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[BC] Device: {device}")

    # ── Load demos ────────────────────────────────────────────────────────────
    print("\n[BC] Loading recordings...")
    dataset = DemoDataset(recordings_dir, skip_noop=skip_noop)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=(device.type == "cuda"))

    # ── Build policy (identical to train.py) ──────────────────────────────────
    dummy_env = DummyVecEnv([lambda: PokemonRedEnv(headless=True)])

    if resume and Path(resume + ".zip").exists():
        print(f"\n[BC] Resuming from {resume}.zip ...")
        model = RecurrentPPO.load(
            resume, env=dummy_env,
            custom_objects={"features_extractor_class": PokemonCnnExtractor,
                            "features_extractor_kwargs": {}},
        )
    else:
        print("\n[BC] Building model (same architecture as train.py)...")
        model = RecurrentPPO(
            policy        = "MultiInputLstmPolicy",
            env           = dummy_env,
            verbose       = 0,
            seed          = 42,
            policy_kwargs = dict(
                features_extractor_class  = PokemonCnnExtractor,
                features_extractor_kwargs = {},
                lstm_hidden_size          = 256,
                n_lstm_layers             = 1,
                shared_lstm               = False,
            ),
        )
    dummy_env.close()

    policy    = model.policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    lstm_hidden = model.policy.lstm_actor.hidden_size

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(f"\n[BC] Training  epochs={n_epochs}  batch={batch_size}  lr={lr}")
    print(f"     {len(dataset):,} demo steps  →  {len(loader):,} batches/epoch\n")

    best_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        policy.train()
        epoch_loss = epoch_acc = n_batches = 0

        for screen, state_vec, minimap, acts in loader:
            screen    = screen.to(device)
            state_vec = state_vec.to(device)
            minimap   = minimap.to(device)
            acts      = acts.to(device)

            n = screen.shape[0]
            obs_th = {"screen": screen, "state": state_vec, "minimap": minimap}

            # Zero LSTM state — every step is treated as the start of a new
            # episode during BC so the LSTM doesn't matter yet.
            h0 = torch.zeros(1, n, lstm_hidden, device=device)
            c0 = torch.zeros_like(h0)
            lstm_states  = RNNStates(pi=(h0, c0), vf=(h0, c0))
            ep_starts    = torch.ones(n, dtype=torch.float32, device=device)

            # evaluate_actions returns (values, log_prob, entropy)
            # log_prob = log P(human_action | obs) under current policy
            _, log_prob, _ = policy.evaluate_actions(obs_th, acts, lstm_states, ep_starts)
            loss = -log_prob.mean()   # negative log-likelihood

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            # Quick accuracy estimate (greedy argmax vs human)
            with torch.no_grad():
                features   = policy.extract_features(obs_th, policy.features_extractor)
                lstm_out, _ = policy.lstm_actor(
                    features.unsqueeze(0),
                    (h0.detach(), c0.detach()),
                )
                latent_pi, _ = policy.mlp_extractor(lstm_out.squeeze(0))
                logits = policy.action_net(latent_pi)
                pred   = logits.argmax(dim=-1)
                acc    = (pred == acts).float().mean().item()

            epoch_loss += loss.item()
            epoch_acc  += acc
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc  = 100.0 * epoch_acc / max(n_batches, 1)
        scheduler.step()

        marker = " ◀ best" if avg_loss < best_loss else ""
        if avg_loss < best_loss:
            best_loss = avg_loss
        print(f"  Epoch {epoch:3d}/{n_epochs}  loss={avg_loss:.4f}  acc={avg_acc:.1f}%{marker}")

    # ── Save as RecurrentPPO zip so train.py --resume can load it ─────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.policy = policy.cpu()
    model.save(output_path)
    print(f"\n[BC] Saved → {output_path}.zip")
    print(f"\n     ┌─ Next step ──────────────────────────────────────────────────")
    print(f"     │  uv run python train.py --resume {output_path}")
    print(f"     └──────────────────────────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Behavioral cloning from human Pokémon recordings.")
    ap.add_argument("--recordings", default="recordings", metavar="DIR",
                    help="Folder with .pkl files from record.py (default: recordings/)")
    ap.add_argument("--output",     default="runs/bc_pretrained", metavar="PATH",
                    help="Output path (no extension) — train.py --resume loads this")
    ap.add_argument("--epochs",     type=int,   default=15,
                    help="Training epochs (default: 15; more = better but diminishing returns)")
    ap.add_argument("--batch-size", type=int,   default=256)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--resume",     default=None, metavar="PATH",
                    help="Resume BC from a previously saved model (no .zip extension)")
    ap.add_argument("--no-noop",    action="store_true",
                    help="Discard no-op steps (cleaner signal if you recorded a lot of idle time)")
    args = ap.parse_args()

    bc_train(
        recordings_dir = args.recordings,
        output_path    = args.output,
        resume         = args.resume,
        n_epochs       = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        skip_noop      = args.no_noop,
    )
