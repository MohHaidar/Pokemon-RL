"""
setup_state.py – Play through the Pokemon Red intro interactively,
then close the PyBoy window to save 'initial_state.state'.

Controls (PyBoy defaults):
    Arrow keys  – D-pad
    Z           – A button
    X           – B button
    Enter       – Start
    Backspace   – Select

Save when:
  ✓ You received your starter Pokemon
  ✓ Prof. Oak gave you the Pokedex
  ✓ Then close the window (click X)

Run once before training:
    python setup_state.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_RENDER_DRIVER", "software")

from pyboy import PyBoy

SAVE_PATH = "initial_state_2.state"

from ram_map import (
    ADDR_PARTY_COUNT      as _PARTY_COUNT,
    ADDR_PARTY_DATA       as _PARTY_DATA,
    PARTY_MON_SIZE        as _PARTY_MON_SIZE,
    ADDR_IN_BATTLE        as _IN_BATTLE,
    ADDR_BADGES           as _BADGES,
    ADDR_MAP_ID           as _MAP_ID,
    ADDR_PLAYER_X         as _PLAYER_X,
    ADDR_PLAYER_Y         as _PLAYER_Y,
    ADDR_ENEMY_SPECIES    as _ENEMY_SPECIES,
    ADDR_ENEMY_LEVEL      as _ENEMY_LEVEL,
    ADDR_ENEMY_STATUS     as _ENEMY_STATUS,
    ADDR_ENEMY_ATK        as _ENEMY_ATK_ADDR,
    ADDR_ENEMY_DEF        as _ENEMY_DEF_ADDR,
    ADDR_ENEMY_SPD        as _ENEMY_SPD_ADDR,
    ADDR_PLAYER_ATK_STAGE as _P_ATK,
    ADDR_PLAYER_DEF_STAGE as _P_DEF,
    ADDR_PLAYER_SPD_STAGE as _P_SPD,
    ADDR_PLAYER_SPC_STAGE as _P_SPC,
    ADDR_ENEMY_ATK_STAGE  as _E_ATK,
    ADDR_ENEMY_DEF_STAGE  as _E_DEF,
    ADDR_ENEMY_SPD_STAGE  as _E_SPD,
    ADDR_ENEMY_SPC_STAGE  as _E_SPC,
    OFF_HP_HI             as _OFF_HP_HI,
    OFF_HP_LO             as _OFF_HP_LO,
    OFF_STATUS            as _OFF_STATUS,
    OFF_LEVEL             as _OFF_LEVEL,
    OFF_MAX_HP_HI         as _OFF_MHP_HI,
    OFF_MAX_HP_LO         as _OFF_MHP_LO,
    OFF_ATK_HI            as _OFF_ATK_HI,
    OFF_ATK_LO            as _OFF_ATK_LO,
    OFF_DEF_HI            as _OFF_DEF_HI,
    OFF_DEF_LO            as _OFF_DEF_LO,
    OFF_SPD_HI            as _OFF_SPD_HI,
    OFF_SPD_LO            as _OFF_SPD_LO,
    STAGE_MULT            as _STAGE_MULT,
    STATUS_NAMES          as _STATUS_NAMES,
    status_mult           as _status_mult,
    status_offense_mult   as _offense_mult,
    status_passive_dmg    as _passive_dmg,
    stage_mult            as _stage_mult,
    fmt_stage             as _fmt_stage,
    status_tag            as _status_tag,
)
# Enemy HP/max-HP addresses — hi byte from ram_map, lo byte = hi + 1
from ram_map import ADDR_ENEMY_HP, ADDR_ENEMY_MAX_HP
_ENEMY_HP_HI  = ADDR_ENEMY_HP
_ENEMY_HP_LO  = ADDR_ENEMY_HP + 1
_ENEMY_MAX_HI = ADDR_ENEMY_MAX_HP
_ENEMY_MAX_LO = ADDR_ENEMY_MAX_HP + 1


def _battle_str(m) -> str:
    """Build a one-line battle strength summary from raw PyBoy memory."""
    # Lead pokemon
    base    = _PARTY_DATA
    p_hp    = (m[base + _OFF_HP_HI]  << 8) | m[base + _OFF_HP_LO]
    p_mhp   = (m[base + _OFF_MHP_HI] << 8) | m[base + _OFF_MHP_LO] or 1
    p_atk   = (m[base + _OFF_ATK_HI] << 8) | m[base + _OFF_ATK_LO]
    p_def   = (m[base + _OFF_DEF_HI] << 8) | m[base + _OFF_DEF_LO]
    p_spd   = (m[base + _OFF_SPD_HI] << 8) | m[base + _OFF_SPD_LO]
    p_lvl   = m[base + _OFF_LEVEL] or 1
    p_st    = m[base + _OFF_STATUS]
    # Enemy
    e_hp    = (m[_ENEMY_HP_HI] << 8) | m[_ENEMY_HP_LO]
    e_mhp   = max((m[_ENEMY_MAX_HI] << 8) | m[_ENEMY_MAX_LO], e_hp, 1)
    e_atk   = (m[_ENEMY_ATK_ADDR] << 8) | m[_ENEMY_ATK_ADDR + 1]
    e_def   = (m[_ENEMY_DEF_ADDR] << 8) | m[_ENEMY_DEF_ADDR + 1]
    e_spd   = (m[_ENEMY_SPD_ADDR] << 8) | m[_ENEMY_SPD_ADDR + 1]
    e_lvl   = m[_ENEMY_LEVEL] or 1
    e_st    = m[_ENEMY_STATUS]
    e_sp    = m[_ENEMY_SPECIES]
    # Stage multipliers
    p_atk_m, p_def_m = _stage_mult(m[_P_ATK]), _stage_mult(m[_P_DEF])
    p_spd_m, p_spc_m = _stage_mult(m[_P_SPD]), _stage_mult(m[_P_SPC])
    e_atk_m, e_def_m = _stage_mult(m[_E_ATK]), _stage_mult(m[_E_DEF])
    e_spd_m, e_spc_m = _stage_mult(m[_E_SPD]), _stage_mult(m[_E_SPC])
    # Raw damage per turn (for display)
    p_dmg   = (p_atk * p_atk_m * _offense_mult(p_st)) / max(e_def * e_def_m, 1)
    e_dmg   = (e_atk * e_atk_m * _offense_mult(e_st)) / max(p_def * p_def_m, 1)
    p_self  = _passive_dmg(p_st, p_mhp)
    e_self  = _passive_dmg(e_st, e_mhp)
    # Base turns-to-kill / turns-to-die
    ttk = e_hp / max(p_dmg + e_self, 0.001)
    ttd = p_hp / max(e_dmg + p_self, 0.001)
    # Conservative adjustments: miss rate, crit risk, turn order
    _HIT  = 0.85
    _CRIT = 0.0625
    adj_ttk = ttk / _HIT
    adj_ttd = ttd / (1.0 + _CRIT)
    p_spd_eff = p_spd * p_spd_m
    e_spd_eff = e_spd * e_spd_m
    if e_spd_eff > p_spd_eff:
        adj_ttd = max(adj_ttd - 1.0, 0.001)   # enemy goes first → player eats extra hit
    elif p_spd_eff > e_spd_eff:
        adj_ttd = adj_ttd + 1.0                # player goes first → enemy misses final-turn attack
    surv    = adj_ttd / max(adj_ttk, 0.001)
    if   surv < 0.5:  verdict = "🚨 OUTMATCHED — RUN"
    elif surv < 0.83: verdict = "⚠ OUTMATCHED"
    elif surv < 1.43: verdict = "~ EVEN"
    else:             verdict = "✓ DOMINANT"
    return (
        f"⚔ LEAD L{p_lvl} {p_hp}/{p_mhp}hp{_status_tag(p_st)} "
        f"ATK{_fmt_stage(m[_P_ATK])} DEF{_fmt_stage(m[_P_DEF])} "
        f"SPD{_fmt_stage(m[_P_SPD])} SPC{_fmt_stage(m[_P_SPC])} dmg→{p_dmg:.1f}"
        f"  |  ENEMY#{e_sp} L{e_lvl} {e_hp}/{e_mhp}hp{_status_tag(e_st)} "
        f"ATK{_fmt_stage(m[_E_ATK])} DEF{_fmt_stage(m[_E_DEF])} "
        f"SPD{_fmt_stage(m[_E_SPD])} SPC{_fmt_stage(m[_E_SPC])} dmg→{e_dmg:.1f}"
        f"  |  ttd={adj_ttd:.1f} ttk={adj_ttk:.1f} surv={surv:.2f} {verdict}"
    )


def main() -> None:
    print("=" * 60)
    print("Pokemon Red – Initial State Setup")
    print("=" * 60)
    print("Controls: Arrow keys | Z=A | X=B | Enter=Start | Bksp=Select")
    print()
    print("SAVE WHEN: starter received + Pokedex from Oak")
    print("  → Close the window (click X) to save and quit")
    print("=" * 60)
    input("Press Enter here to launch the game…")

    pyboy = PyBoy("Pokemon_Red.gb", window="SDL2")
    pyboy.set_emulation_speed(1)

    print("\n[Game is running] — close the window when ready to save.\n")

    frame      = 0
    last_map   = -1
    in_battle  = False
    prev_lines = 1

    while pyboy.tick(1, True):
        frame += 1
        m      = pyboy.memory
        map_id = m[_MAP_ID]
        x      = m[_PLAYER_X]
        y      = m[_PLAYER_Y]
        battle = m[_IN_BATTLE] > 0

        changed = (map_id != last_map) or (battle != in_battle) or (frame % 60 == 0)
        if changed:
            party   = m[_PARTY_COUNT]
            badges  = m[_BADGES]
            line1   = (f"map={map_id:>3}  x={x:>3}  y={y:>3}  "
                       f"party={party}  badges={bin(badges).count('1')}")
            if battle and party > 0:
                line2 = _battle_str(m)
                output = f"\r  {line1}\n  {line2}\033[K\033[1A"
                cur_lines = 2
            else:
                output = f"\r  {line1}\033[K"
                cur_lines = 1
            # Clear extra lines if we had more before
            if prev_lines > cur_lines:
                import sys
                sys.stdout.write("\n\033[K" * (prev_lines - cur_lines))
                sys.stdout.write(f"\033[{prev_lines - cur_lines}A")
            print(output, end="", flush=True)
            last_map   = map_id
            in_battle  = battle
            prev_lines = cur_lines

    print("\nSaving state…")
    with open(SAVE_PATH, "wb") as f:
        pyboy.save_state(f)

    m = pyboy.memory
    print(f"  map={m[_MAP_ID]}  party={m[_PARTY_COUNT]}  badges={m[_BADGES]}")
    print(f"Done! State saved to '{SAVE_PATH}'.")
    print("You can now run:  python train.py")
    pyboy.stop()


if __name__ == "__main__":
    main()

