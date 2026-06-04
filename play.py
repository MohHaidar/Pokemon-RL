"""
play.py - Watch a trained Pokemon Red agent play in real time.

Usage
-----
    python play.py runs/pokemon_ppo_final.zip
    python play.py runs/checkpoints/pokemon_ppo_500000_steps.zip --speed 2
    python play.py runs/pokemon_ppo_final.zip --state initial_state.state

Close the PyBoy window or press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from sb3_contrib import RecurrentPPO

from env import (
    PokemonRedEnv, LOW_HP_THRESHOLD,
)
from ram_map import POKECENTER_MAPS, STAGE_MULT as _STAGE_MULT, NEUTRAL_STAGE as _NEUTRAL_STAGE

_stop = False
def _sigint_handler(sig, frame):
    global _stop
    _stop = True
signal.signal(signal.SIGINT, _sigint_handler)


def _fmt_stage(raw: int) -> str:
    """Format a raw Gen-1 stage value (1-13, neutral=7) as +N / -N / ±0."""
    if not (1 <= raw <= 13):
        raw = _NEUTRAL_STAGE
    s = raw - _NEUTRAL_STAGE
    return f"+{s}" if s > 0 else ("±0" if s == 0 else str(s))


def _stage_mult_of(raw: int) -> float:
    # Clamp to valid range 1-13; treat out-of-range as neutral (7)
    if not (1 <= raw <= 13):
        raw = _NEUTRAL_STAGE
    return _STAGE_MULT[raw - 1]


def _strength_line(label: str, mon: dict, stages: dict, dmg_per_turn: float) -> str:
    """Build a compact per-side display string."""
    hp_r    = min(mon["hp"] / max(mon["max_hp"], mon["hp"], 1), 1.0)
    lvl     = mon["level"]
    atk_raw = stages.get("atk", _NEUTRAL_STAGE)
    def_raw = stages.get("def", _NEUTRAL_STAGE)
    spd_raw = stages.get("spd", _NEUTRAL_STAGE)
    spc_raw = stages.get("spc", _NEUTRAL_STAGE)
    stag    = ""
    status_names = {0x07: "SLP", 0x20: "FRZ", 0x40: "PAR", 0x10: "BRN", 0x08: "PSN"}
    for mask, name in status_names.items():
        if mon.get("status", 0) & mask:
            stag = f"[{name}]"
            break
    return (
        f"{label} L{lvl} {hp_r:.0%}hp{stag} "
        f"ATK{_fmt_stage(atk_raw)} DEF{_fmt_stage(def_raw)} "
        f"SPD{_fmt_stage(spd_raw)} SPC{_fmt_stage(spc_raw)} "
        f"dmg→{dmg_per_turn:.1f}"
    )


def evaluate(
    model_path: str,
    rom_path: str = "Pokemon_Red.gb",
    n_episodes: int = 10,
    max_steps: int = 8_192,
) -> None:
    """Run N headless episodes and report true mean/min/max reward."""
    if not Path(model_path).exists():
        sys.exit(f"[eval] Model not found: {model_path}")

    env   = PokemonRedEnv(rom_path=rom_path, headless=True, max_steps=max_steps)
    model = RecurrentPPO.load(model_path)

    print(f"[eval] Evaluating {n_episodes} episodes (headless)...")
    rewards, tiles, badges_list = [], [], []

    for ep in range(n_episodes):
        obs, _         = env.reset()
        lstm_states    = None
        episode_start  = True
        ep_reward      = 0.0

        while True:
            action, lstm_states = model.predict(
                obs, state=lstm_states,
                episode_start=episode_start, deterministic=False,
            )
            obs, r, terminated, truncated, info = env.step(int(action))
            episode_start = terminated or truncated
            ep_reward    += r
            if terminated or truncated:
                break

        state = info.get("state", {})
        rewards.append(ep_reward)
        tiles.append(len(env._visited_coords))
        badges_list.append(bin(state.get("badges", 0)).count("1"))
        print(f"  ep {ep+1:>2}: reward={ep_reward:>8.2f}  tiles={tiles[-1]:>4}  badges={badges_list[-1]}")

    env.close()
    print(f"\n[eval] Mean reward : {sum(rewards)/len(rewards):.2f}")
    print(f"[eval] Min  reward : {min(rewards):.2f}")
    print(f"[eval] Max  reward : {max(rewards):.2f}")
    print(f"[eval] Mean tiles  : {sum(tiles)/len(tiles):.1f}")
    print(f"[eval] Mean badges : {sum(badges_list)/len(badges_list):.2f}")


def play(
    model_path:     str,
    rom_path:       str = "Pokemon_Red.gb",
    state_path:     str | None = None,
    speed:          int = 1,
    display_frames: int = 8,
    max_steps:      int = 8_192,
) -> None:
    """Watch the agent play using the native PyBoy SDL2 window.

    Menus, text boxes and battle UI now render correctly via the patched
    wy_activated_frame fix compiled into PyBoy from source.

    speed=1 → real-time (~60 fps).  Increase for faster playback.
    display_frames: idle frames run after each agent action so
    menus/text/animations are visible before the next action.
    Close the window or press Ctrl+C to quit.
    """
    if not Path(model_path).exists():
        sys.exit(f"[play] Model not found: {model_path}")

    env = PokemonRedEnv(
        rom_path        = rom_path,
        headless        = False,
        emulation_speed = speed,
        display_frames  = display_frames,
        max_steps       = max_steps,
    )

    print(f"[play] Loading model from {model_path}...")
    model = RecurrentPPO.load(model_path, n_envs=1, device="cpu")

    # Compatibility: older checkpoints may not include 'minimap'.
    _model_obs_keys = set(model.observation_space.spaces.keys())

    obs, _ = env.reset()

    if state_path:
        with open(state_path, "rb") as f:
            env._pyboy.load_state(f)
        obs, _, _, _, _ = env.step(0)

    episode_reward = 0.0
    episode_steps  = 0
    episode        = 1
    lstm_states    = None
    episode_start  = True
    _prev_n_lines  = 2

    print("[play] Agent is playing... Close the window or Ctrl+C to quit.\n")
    header = f"{'Step':>6}  {'Reward':>10}  {'Lvl':>4}  {'Bdg':>3}  {'Party':>14}  {'Balls':>5}  {'Map':>3}  {'Gyms'}"
    print(header)
    print("-" * len(header))

    # Episode-level debug accumulators
    ep_low_hp      = 0.0
    ep_crit_battle = 0.0
    ep_nurse_prox  = 0.0
    steps_since_heal = 0
    _needs_clear   = False   # whether to overwrite the previous 2 display lines

    while not _stop:
        try:
            action, lstm_states = model.predict(
                {k: v for k, v in obs.items() if k in _model_obs_keys},
                state          = lstm_states,
                episode_start  = episode_start,
                deterministic  = False,
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
        except Exception:
            break  # SDL2 window closed

        episode_start   = terminated or truncated
        episode_reward += float(reward)
        episode_steps  += 1

        state = info.get("state", {})
        party = state.get("party", [])
        items = state.get("items", {})
        enemy = state.get("enemy")
        bd    = info.get("reward_breakdown", {})

        # ── Accumulate debug stats ────────────────────────────────────────
        ep_low_hp      += bd.get("low_hp",          0.0)
        ep_crit_battle += bd.get("critical_battle",  0.0)
        ep_nurse_prox  += bd.get("nurse_nav", 0.0) + bd.get("nurse_stand", 0.0) + bd.get("nurse_a", 0.0)
        if "pokecenter_heal" in bd:
            steps_since_heal = 0
        else:
            steps_since_heal += 1

        # ── Line 1: main status ───────────────────────────────────────────
        party_str = " ".join(f"L{p['level']}" for p in party) or "—"
        balls     = items.get("balls", 0)
        gyms      = len(getattr(env, "_visited_gyms", set()))
        badges    = bin(state.get("badges", 0)).count("1")
        in_battle = " [BATTLE]" if state.get("in_battle") else ""

        moves_str = ""
        if party:
            moves_str = " moves:" + "/".join(
                str(m) for m in party[0].get("moves", []) if m != 0
            )

        enemy_str = ""
        if enemy:
            e_ratio = enemy["hp"] / max(enemy["max_hp"], 1)
            enemy_str = f" vs#{enemy['species']}({e_ratio:.0%})"

        _QUIET = {"critical_battle", "low_hp", "nurse_nav"}
        events = [
            f"{'+' if v > 0 else ''}{v:.1f} {k}"
            for k, v in bd.items() if k not in _QUIET
        ]
        event_str = f"  [{', '.join(events)}]" if events else ""

        line1 = (
            f"{episode_steps:>6}  {episode_reward:>10.2f}"
            f"  {party[0]['level'] if party else '?':>4}"
            f"  {badges:>3}  {party_str:<14}  {balls:>5}"
            f"  {state.get('map_id','?'):>3}  {gyms}gyms"
            f"{moves_str}{enemy_str}{in_battle}{event_str}"
        )

        # ── Line 2: HP / healing debug ────────────────────────────────────
        if party:
            lead     = party[0]
            hp_now   = lead["hp"]
            hp_max   = lead["max_hp"]
            hp_ratio = hp_now / max(hp_max, 1)
            bar_full = int(hp_ratio * 10)
            hp_bar   = "█" * bar_full + "░" * (10 - bar_full)
            if hp_ratio < LOW_HP_THRESHOLD:
                hp_tag = " ⚠LOW"
            elif hp_ratio < 0.50:
                hp_tag = " low"
            else:
                hp_tag = ""
            hp_str = f"HP [{hp_bar}]{hp_now}/{hp_max}({hp_ratio:.0%}){hp_tag}"
        else:
            hp_str = "HP [??????????]"

        in_pc    = state.get("map_id") in POKECENTER_MAPS
        pc_tag   = " [IN-PC]" if in_pc else ""
        heal_tag = f"  heal_in:{steps_since_heal}s" if steps_since_heal < 9999 else ""

        nurse_now = bd.get("nurse_nav", 0.0)
        nurse_str = f"  prox:{nurse_now:+.2f}" if in_pc else ""

        dex_count = len(state.get("pokedex_owned", set()))
        dex_str   = f"  dex:{dex_count}/151"

        line2 = (
            f"  {hp_str}{pc_tag}{nurse_str}"
            f"  | ∑low_hp:{ep_low_hp:+.1f}"
            f"  ∑crit:{ep_crit_battle:+.1f}"
            f"  ∑nurse:{ep_nurse_prox:+.1f}"
            f"{dex_str}"
            f"{heal_tag}"
        )

        # ── Line 3: strength breakdown when in battle ─────────────────────
        battle_line = ""
        if state.get("in_battle") and party and enemy:
            stages   = state.get("stages", {})
            ps       = stages.get("player", {})
            es       = stages.get("enemy",  {})
            p_atk_m  = _stage_mult_of(ps.get("atk", _NEUTRAL_STAGE))
            p_def_m  = _stage_mult_of(ps.get("def", _NEUTRAL_STAGE))
            p_spd_m  = _stage_mult_of(ps.get("spd", _NEUTRAL_STAGE))
            e_atk_m  = _stage_mult_of(es.get("atk", _NEUTRAL_STAGE))
            e_def_m  = _stage_mult_of(es.get("def", _NEUTRAL_STAGE))
            e_spd_m  = _stage_mult_of(es.get("spd", _NEUTRAL_STAGE))
            lead     = party[0]
            from env import gen1_dmg_per_turn, combat_survivability
            p_dmg_turn = gen1_dmg_per_turn(
                lead["level"], lead["atk_stat"], p_atk_m, lead.get("status", 0),
                enemy["def_stat"], e_def_m, enemy.get("status", 0), enemy["max_hp"])
            e_dmg_turn = gen1_dmg_per_turn(
                enemy["level"], enemy["atk_stat"], e_atk_m, enemy.get("status", 0),
                lead["def_stat"], p_def_m, lead.get("status", 0), lead["max_hp"])
            ttd, ttk, surv = combat_survivability(
                lead["hp"],  p_dmg_turn, lead.get("spd_stat", 10)  * p_spd_m,
                enemy["hp"], e_dmg_turn, enemy.get("spd_stat", 10) * e_spd_m,
            )
            hp_ratio = lead["hp"] / max(lead["max_hp"], 1)
            # Adjust verdict when at critical HP — even "even" fights are dangerous
            if hp_ratio < LOW_HP_THRESHOLD:
                verdict = "🚨 LOW HP — RUN"
            elif surv < 0.5:
                verdict = "🚨 OUTMATCHED — RUN"
            elif surv < 0.83:
                verdict = "⚠ OUTMATCHED"
            elif surv < 1.43:
                verdict = "~ EVEN"
            else:
                verdict = "✓ DOMINANT"
            lead_str  = _strength_line("LEAD", lead, ps, p_dmg_turn)
            enemy_str = _strength_line(f"ENEMY#{enemy.get('species','?')}", enemy, es, e_dmg_turn)
            battle_line = (
                f"  ⚔ {lead_str}  |  {enemy_str}"
                f"  |  ttd={ttd:.1f} ttk={ttk:.1f} surv={surv:.2f} {verdict}"
            )

        # ── Render: overwrite previous lines each step ────────────────────
        n_lines = 3 if battle_line else 2
        if _needs_clear:
            sys.stdout.write(f"\033[{_prev_n_lines}A\033[J")
        output = line1 + "\n" + line2 + "\n"
        if battle_line:
            output += battle_line + "\n"
        sys.stdout.write(output)
        sys.stdout.flush()
        _needs_clear   = True
        _prev_n_lines  = n_lines

        if terminated or truncated:
            _needs_clear = False
            ep_bd = info.get("ep_breakdown", {})
            ep_lines = "  ".join(
                f"{k}: {v:+.1f}" for k, v in sorted(ep_bd.items(), key=lambda x: -abs(x[1]))
            )
            print(f"[ep {episode}] reward={episode_reward:.2f}  steps={episode_steps}")
            print(f"  breakdown → {ep_lines}")
            episode       += 1
            episode_reward = 0.0
            episode_steps  = 0
            lstm_states    = None
            episode_start  = True
            ep_low_hp      = 0.0
            ep_crit_battle = 0.0
            ep_nurse_prox  = 0.0
            steps_since_heal = 0
            obs, _         = env.reset()

    print("\n[play] Stopped.")
    try:
        env.close()
    except OSError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--rom",   default="Pokemon_Red.gb")
    parser.add_argument("--state", default=None)
    parser.add_argument("--speed",   type=int, default=1,
                        help="Emulation speed: 1=real-time (default), 2=2x, 0=unlimited")
    parser.add_argument("--display", type=int, default=8,
                        help="Extra idle frames after each action so menus/text are visible "
                             "(default: 8; increase if text disappears too fast)")
    parser.add_argument("--eval",  type=int, default=0,
                        help="Run N headless evaluation episodes instead of watching")
    parser.add_argument("--max-steps", type=int, default=8_192,
                        help="Episode length cap (default: 8192)")
    args = parser.parse_args()

    if args.eval > 0:
        evaluate(model_path=args.model, rom_path=args.rom, n_episodes=args.eval,
                 max_steps=args.max_steps)
    else:
        play(
            model_path     = args.model,
            rom_path       = args.rom,
            state_path     = args.state,
            speed          = args.speed,
            display_frames = args.display,
            max_steps      = args.max_steps,
        )

