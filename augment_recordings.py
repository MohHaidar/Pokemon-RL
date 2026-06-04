#!/usr/bin/env python3
"""
augment_recordings.py — Augment human recordings into many heal-focused BC examples.

For each recording, finds every heal event (pokecenter_heal / pokecenter_heal_once)
and generates many subtrajectories that all end at the heal.  This gives the policy
dense examples of "go to the PC and press A" from many different starting distances.

Augmentation strategies applied per heal event:
  1. Sliding-window crops  — same heal path, different start distances
  2. State-vector noise    — small Gaussian noise on continuous state features
                             (clips to [0, 1] — binary flags stay intact)
  3. Action noise          — randomly replace ~3 % of noop steps with a
                             directional (encourages exploration during BC)

Output .pkl files are compatible with bc_pretrain.py directly.

Disk-space note:
  Screen observations dominate size (~28 KB/step).  The default max-window of 300
  steps means each augmented trajectory is at most ~8 MB.  With default settings:
  ~24 heal events × 4 windows × 2 noise copies = ~192 trajectories ≈ 1–2 GB output.
  Raise --max-window or lower --stride for more data, lower them to save space.

Usage:
    uv run python augment_recordings.py                        # defaults
    uv run python augment_recordings.py --input recordings/heal_demos \\
        --output recordings/augmented --max-window 500 --noise-copies 3
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np

# ── Continuous state-vector indices (safe to perturb with Gaussian noise) ─────
# Binary / integer-coded indices are excluded so noise doesn't corrupt flags.
# State is 91-dim: indices 0-56 (obs features), 57-90 (combat stats — all continuous).
_BINARY_INDICES: frozenset[int] = frozenset({23, 25, 26, 31, 36, 37, 38, 41, 53})
_STATE_DIM = 91
_CONTINUOUS_MASK = np.array(
    [i not in _BINARY_INDICES for i in range(_STATE_DIM)], dtype=bool
)

# Directional actions (used for action noise replacement of noops)
_DIRECTIONAL = np.array([1, 2, 3, 4], dtype=np.int64)


def _find_heal_steps(infos: list[dict]) -> list[int]:
    """Return step indices where a pokecenter heal reward was granted."""
    heal_steps = []
    for i, bd in enumerate(infos):
        if bd.get("pokecenter_heal", 0.0) > 0 or bd.get("pokecenter_heal_once", 0.0) > 0:
            heal_steps.append(i)
    return heal_steps


def _make_subtrajectory(
    obs: dict,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    infos: list[dict],
    start: int,
    end: int,          # inclusive — last step kept
) -> dict:
    """Slice a trajectory [start:end+1] and mark the final step as done."""
    sliced_obs = {k: v[start : end + 1] for k, v in obs.items()}
    sliced_acts = actions[start : end + 1].copy()
    sliced_rews = rewards[start : end + 1].copy()
    sliced_done = dones[start : end + 1].copy()
    sliced_info = infos[start : end + 1]

    sliced_done[-1] = True   # treat heal as episode end for BC
    return {
        "observations": sliced_obs,
        "actions":      sliced_acts,
        "rewards":      sliced_rews,
        "dones":        sliced_done,
        "infos":        sliced_info,
    }


def _add_noise(traj: dict, noise_std: float, action_noise_p: float, rng: np.random.Generator) -> dict:
    """Return a copy of traj with state noise and light action noise."""
    traj = dict(traj)
    obs = {k: v.copy() for k, v in traj["observations"].items()}

    # State noise — continuous features only
    state = obs["state"].copy().astype(np.float32)
    noise = rng.normal(0.0, noise_std, size=state.shape).astype(np.float32)
    noise[:, ~_CONTINUOUS_MASK] = 0.0
    state = np.clip(state + noise, 0.0, 1.0)
    obs["state"] = state

    # Action noise — replace a fraction of noop steps with a random direction
    acts = traj["actions"].copy()
    noop_mask = acts == 0
    flip = rng.random(len(acts)) < action_noise_p
    replace_mask = noop_mask & flip
    if replace_mask.any():
        acts[replace_mask] = rng.choice(_DIRECTIONAL, size=replace_mask.sum())

    traj["observations"] = obs
    traj["actions"] = acts
    return traj


def augment_file(
    path: str,
    output_dir: str,
    stride: int,
    min_window: int,
    max_window: int,
    noise_copies: int,
    noise_std: float,
    action_noise_p: float,
    rng: np.random.Generator,
) -> int:
    """Augment one recording file.  Returns number of new trajectories saved."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    obs     = data["observations"]          # dict of arrays
    actions = data["actions"]               # (N,) int64
    rewards = data["rewards"]               # (N,) float32
    dones   = data["dones"]                 # (N,) bool
    infos   = data["infos"]                 # list[dict]  (reward breakdowns)

    heal_steps = _find_heal_steps(infos)
    if not heal_steps:
        print(f"  [skip] No heal event found in {os.path.basename(path)} — skipping")
        return 0

    stem = Path(path).stem
    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for heal_idx in heal_steps:
        # End a few steps after the heal so the "healed" observation is included
        end = min(heal_idx + 5, len(actions) - 1)

        # Clamp start range to max_window steps before heal
        earliest_start = max(0, heal_idx - max_window)

        # Generate windows ending at heal_idx with different start points
        starts = list(range(earliest_start, heal_idx - min_window + 1, stride))
        if not starts:
            starts = [earliest_start]  # window shorter than min_window — use max available

        for start in starts:
            base_traj = _make_subtrajectory(obs, actions, rewards, dones, infos, start, end)

            # Save the clean (no noise) version
            out_path = os.path.join(
                output_dir, f"{stem}_heal{heal_idx}_s{start:04d}_n0.pkl"
            )
            with open(out_path, "wb") as f:
                pickle.dump(base_traj, f, protocol=pickle.HIGHEST_PROTOCOL)
            saved += 1

            # Save noise copies
            for copy_idx in range(1, noise_copies + 1):
                noisy = _add_noise(base_traj, noise_std, action_noise_p, rng)
                out_path = os.path.join(
                    output_dir, f"{stem}_heal{heal_idx}_s{start:04d}_n{copy_idx}.pkl"
                )
                with open(out_path, "wb") as f:
                    pickle.dump(noisy, f, protocol=pickle.HIGHEST_PROTOCOL)
                saved += 1

    return saved


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Augment heal recordings into many BC training examples."
    )
    ap.add_argument("--input",        default="recordings",           metavar="DIR",
                    help="Directory with source .pkl recordings (default: recordings/)")
    ap.add_argument("--output",       default="recordings/augmented", metavar="DIR",
                    help="Output directory for augmented files (default: recordings/augmented/)")
    ap.add_argument("--stride",       type=int,   default=60,
                    help="Steps between sliding-window start points (default: 60)")
    ap.add_argument("--min-window",   type=int,   default=50,
                    help="Minimum trajectory length kept (default: 50)")
    ap.add_argument("--max-window",   type=int,   default=300,
                    help="Maximum steps before heal to include (default: 300, ~8 MB/trajectory)")
    ap.add_argument("--noise-copies", type=int,   default=2,
                    help="Number of noise-augmented copies per window (default: 2)")
    ap.add_argument("--noise-std",    type=float, default=0.02,
                    help="Gaussian noise std on state features (default: 0.02)")
    ap.add_argument("--action-noise", type=float, default=0.03,
                    help="Probability of replacing a noop with a directional (default: 0.03)")
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    rng   = np.random.default_rng(args.seed)
    paths = sorted(glob.glob(os.path.join(args.input, "*.pkl")))

    if not paths:
        print(f"No .pkl files found in '{args.input}/'. Record some gameplay first:")
        print("  uv run python record.py --save-dir recordings/heal_demos")
        return

    print(f"[augment] Found {len(paths)} recording(s) in '{args.input}/'")
    print(f"  stride={args.stride}  min_window={args.min_window}  max_window={args.max_window}  "
          f"noise_copies={args.noise_copies}  noise_std={args.noise_std}")
    print(f"  -> output: '{args.output}/'")
    print()

    total_saved = 0
    for path in paths:
        n = augment_file(
            path, args.output,
            stride=args.stride,
            min_window=args.min_window,
            max_window=args.max_window,
            noise_copies=args.noise_copies,
            noise_std=args.noise_std,
            action_noise_p=args.action_noise,
            rng=rng,
        )
        print(f"  ok  {os.path.basename(path)} -> {n} augmented trajectories")
        total_saved += n

    print(f"\n[augment] Done: {total_saved} total trajectories saved to '{args.output}/'")
    print(f"\n  Next step:")
    print(f"  uv run python bc_pretrain.py --recordings {args.output}")


if __name__ == "__main__":
    main()
