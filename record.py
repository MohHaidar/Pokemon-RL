#!/usr/bin/env python3
"""
record.py — Human play recorder for imitation learning.

The game runs at normal speed in the SDL2 window.  Your keyboard drives
the Game Boy directly.  Rewards and penalties are printed live in the
terminal so you can see exactly what the bot would earn from your play.

Each episode is saved to  recordings/<timestamp>_ep<N>.pkl
Format:
    {
        'observations': float32 array (N, obs_shape),
        'actions':      int64   array (N,),
        'rewards':      float32 array (N,),
        'dones':        bool    array (N,),
        'infos':        list of reward-breakdown dicts,
    }

Controls (active when the SDL2 window is focused):
    Arrow keys  → move
    A           → A button  (confirm / attack)
    S           → B button  (cancel / bag)
    Enter       → Start menu
    Right Shift → Select

Usage:
    uv run python record.py                   # unlimited episodes
    uv run python record.py --episodes 5      # stop after 5 episodes
    uv run python record.py --save-dir demos  # custom directory
"""

import argparse
import os
import pickle
import sys
from datetime import datetime
from threading import Lock

import numpy as np
from pynput import keyboard as pynput_kb
from pynput.keyboard import Key

# ── make sure the env module is importable ────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from env import PokemonRedEnv, ACTIONS, IDLE_FRAMES, PLAY_FRAME_SKIP

# ── Key → action-index mapping ────────────────────────────────────────────────
# Action indices match env.py: 0=noop, 1=up, 2=down, 3=left, 4=right,
#                               5=a,   6=b,   7=start, 8=select
KEY_MAP: dict = {
    Key.up:       1,
    Key.down:     2,
    Key.left:     3,
    Key.right:    4,
    "a":          5,   # A button
    "s":          6,   # B button
    Key.enter:    7,   # Start
    Key.shift_r:  8,   # Select
}

# ── Thread-safe key-state tracker ─────────────────────────────────────────────
_pressed: set = set()
_lock = Lock()


def _on_press(key):
    k = key.char.lower() if (hasattr(key, "char") and key.char) else key
    with _lock:
        _pressed.add(k)


def _on_release(key):
    k = key.char.lower() if (hasattr(key, "char") and key.char) else key
    with _lock:
        _pressed.discard(k)


def detect_action() -> int:
    """Return the highest-priority action from the current key state."""
    with _lock:
        snapshot = _pressed.copy()
    for key, action in KEY_MAP.items():
        if key in snapshot:
            return action
    return 0  # no-op


# ── Terminal display helpers ───────────────────────────────────────────────────
_COL_W = 130  # terminal columns reserved for display lines

def _fmt_breakdown(bd: dict, reward: float, steps: int,
                   ep_reward: float, totals: dict, info: dict) -> str:
    """Three-line live display: status + this-step events + all running totals."""
    state = info.get("state", {})
    party = state.get("party", [])
    in_battle = state.get("in_battle", False)

    # ── Line 1: context bar ────────────────────────────────────────────────
    status = "[BATTLE]" if in_battle else f"map:{state.get('map_id','?')}"
    lead_hp = ""
    if party:
        p = party[0]
        ratio = p['hp'] / max(p['max_hp'], 1)
        bar = "█" * int(ratio * 10) + "░" * (10 - int(ratio * 10))
        n_alive = sum(1 for x in party if x.get("hp", 0) > 0)
        lead_hp = f"  HP:[{bar}]{p['hp']}/{p['max_hp']}  party:{n_alive}/{len(party)}alive"
    badges = bin(state.get("badges", 0)).count("1")
    line1 = f"  [{steps:5d}] R={reward:+6.2f}  Σ={ep_reward:+7.2f}  {status}{lead_hp}  badges:{badges}"

    # ── Line 2: this step's nonzero components ────────────────────────────
    pos = "  ".join(f"{k}:+{v:.3f}" for k, v in sorted(bd.items(), key=lambda x: -x[1]) if v > 0)
    neg = "  ".join(f"{k}:{v:.3f}"  for k, v in sorted(bd.items(), key=lambda x:  x[1]) if v < 0)
    step_parts = []
    if pos: step_parts.append(f"▲ {pos}")
    if neg: step_parts.append(f"▼ {neg}")
    line2 = "  THIS STEP: " + ("  |  ".join(step_parts) if step_parts else "(no reward)")

    # ── Line 3: all running totals sorted by abs value ────────────────────
    if totals:
        all_pos = "  ".join(
            f"{k}:{v:+.1f}" for k, v in sorted(totals.items(), key=lambda x: -x[1]) if v > 0
        )
        all_neg = "  ".join(
            f"{k}:{v:+.1f}" for k, v in sorted(totals.items(), key=lambda x:  x[1]) if v < 0
        )
        parts3 = []
        if all_pos: parts3.append(f"▲ {all_pos}")
        if all_neg: parts3.append(f"▼ {all_neg}")
        line3 = "  TOTALS:   " + "  |  ".join(parts3)
    else:
        line3 = "  TOTALS:   (none yet)"

    return line1, line2, line3


# ── Trajectory save ───────────────────────────────────────────────────────────

def save_trajectory(traj: list, save_dir: str, episode: int) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"{ts}_ep{episode:03d}.pkl")

    # observations is a dict — stack each key separately
    obs_keys = traj[0][0].keys()
    obs   = {k: np.array([t[0][k] for t in traj]) for k in obs_keys}
    acts  = np.array([t[1] for t in traj], dtype=np.int64)
    rews  = np.array([t[2] for t in traj], dtype=np.float32)
    dones = np.array([t[3] for t in traj], dtype=bool)
    infos = [t[4] for t in traj]

    with open(path, "wb") as f:
        pickle.dump(
            {"observations": obs, "actions": acts, "rewards": rews,
             "dones": dones, "infos": infos},
            f, protocol=pickle.HIGHEST_PROTOCOL,
        )
    return path


# ── Main recording loop ───────────────────────────────────────────────────────

def run(save_dir: str = "recordings", max_episodes: int | None = None, speed: int = 1):
    env = PokemonRedEnv(headless=False, emulation_speed=speed, play_frame_skip=16)

    # Start keyboard listener (background thread, no window focus needed)
    listener = pynput_kb.Listener(on_press=_on_press, on_release=_on_release, suppress=False)
    listener.start()

    episode      = 0
    total_steps  = 0

    print(__doc__)
    print("=" * _COL_W)
    print("  Recording started.  Close the SDL2 window or press Ctrl+C to stop.")
    print("=" * _COL_W)

    try:
        while max_episodes is None or episode < max_episodes:
            obs, _ = env.reset()
            traj: list = []
            ep_reward  = 0.0
            ep_steps   = 0
            done = truncated = False
            totals: dict = {}   # running reward-component totals
            _prev_lines = 0     # how many lines to erase on next update

            ep_label = f"Episode {episode + 1}"
            if max_episodes:
                ep_label += f" / {max_episodes}"
            print(f"\n  ▶ {ep_label}")

            while not done and not truncated and env.window_open:
                # ── detect what the human is pressing RIGHT NOW ──────────────
                action = detect_action()

                # Tell the env which action was taken so reward gating works
                # correctly (e.g. wall-penalty only for directional, dialogue
                # reward excludes Start/Select).  We pass action=0 to env.step
                # so PyBoy does NOT inject a button programmatically — the SDL2
                # window already handles the human's keypresses directly.
                env._record_action_override = action

                obs_next, reward, done, truncated, info = env.step(0)

                ep_reward += reward
                ep_steps  += 1
                total_steps += 1

                traj.append((obs, action, reward, done or truncated,
                             info.get("reward_breakdown", {})))
                obs = obs_next

                # Accumulate totals
                bd = info.get("reward_breakdown", {})
                for k, v in bd.items():
                    totals[k] = totals.get(k, 0.0) + v

                # Live reward display — always update (erase previous 3 lines)
                line1, line2, line3 = _fmt_breakdown(bd, reward, ep_steps, ep_reward, totals, info)
                if _prev_lines:
                    sys.stdout.write(f"\033[{_prev_lines}A\033[J")
                sys.stdout.write(f"{line1}\n{line2}\n{line3}\n")
                sys.stdout.flush()
                _prev_lines = 3

            # ── episode ended ─────────────────────────────────────────────────
            _prev_lines = 0
            if traj:
                path = save_trajectory(traj, save_dir, episode + 1)
                print(f"\n  ✓ {ep_label}: total_reward={ep_reward:.2f}  steps={ep_steps}")
                # Print sorted full breakdown
                top = sorted(totals.items(), key=lambda x: -abs(x[1]))
                print("  Final breakdown:")
                for k, v in top:
                    bar = "▲" if v > 0 else "▼"
                    print(f"    {bar} {k:<22} {v:+.2f}")
                print(f"    Saved → {path}")
            else:
                print(f"\n  (Episode {episode + 1} was empty, not saved)")

            episode += 1

            if not env.window_open:
                print("\n  SDL2 window closed — stopping.")
                break

    except KeyboardInterrupt:
        print("\n  Ctrl+C — stopping.")
    finally:
        listener.stop()
        env.close()

    print(f"\n  Done: {episode} episode(s), {total_steps} total steps.")
    print(f"  Recordings saved in: {os.path.abspath(save_dir)}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Record human Pokémon Red play for imitation learning.")
    ap.add_argument("--save-dir",  default="recordings", metavar="DIR",
                    help="Directory to save trajectory files (default: recordings/)")
    ap.add_argument("--episodes",  type=int, default=None, metavar="N",
                    help="Stop after N episodes (default: run until window is closed)")
    ap.add_argument("--speed",     type=int, default=1, metavar="N",
                    help="Emulation speed: 1=real-time (default), 2=2x, 0=unlimited")
    args = ap.parse_args()
    run(save_dir=args.save_dir, max_episodes=args.episodes, speed=args.speed)
