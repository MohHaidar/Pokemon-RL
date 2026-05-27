"""
Unit tests for game_helpers.py — pure QUERY layer functions.
No ROM or emulator needed; all tests use synthetic state dicts.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
from ram_map import POKECENTER_MAPS, NEUTRAL_STAGE
from game_helpers import (
    is_in_pokecenter, is_in_battle, is_trainer_battle, is_wild_battle,
    is_text_box_open, get_lead, get_lead_hp_ratio, count_alive_pokemon,
    count_badges, get_enemy, get_enemy_hp, get_enemy_hp_ratio,
    get_enemy_stats, get_player_stages, get_enemy_stages,
    enemy_stat_lowered, get_player_pos,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_party(*hp_max_pairs):
    """Build a minimal party list from (hp, max_hp) pairs."""
    return [{"hp": hp, "max_hp": mx, "level": 5, "status": 0} for hp, mx in hp_max_pairs]


def _make_enemy(hp=30, max_hp=30, level=5, status=0,
                atk_stat=10, def_stat=10, spd_stat=10):
    return {
        "hp": hp, "max_hp": max_hp, "level": level, "status": status,
        "atk_stat": atk_stat, "def_stat": def_stat, "spd_stat": spd_stat,
    }


def _neutral_stages():
    return {k: NEUTRAL_STAGE for k in ("atk", "def", "spd", "spc", "acc", "eva")}


# ── map / position ──────────────────────────────────────────────────────────

def test_get_player_pos():
    state = {"player_x": 7, "player_y": 3}
    assert get_player_pos(state) == (7, 3)


def test_is_in_pokecenter_true():
    pc_id = next(iter(POKECENTER_MAPS))
    assert is_in_pokecenter({"map_id": pc_id})


def test_is_in_pokecenter_false():
    assert not is_in_pokecenter({"map_id": 0})  # Pallet Town, not a PC


# ── battle flags ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("val,exp", [(0, False), (1, True), (2, True)])
def test_is_in_battle(val, exp):
    assert is_in_battle({"in_battle": val}) == exp


def test_is_trainer_battle():
    assert is_trainer_battle({"in_battle": 2})
    assert not is_trainer_battle({"in_battle": 1})


def test_is_wild_battle():
    assert is_wild_battle({"in_battle": 1})
    assert not is_wild_battle({"in_battle": 2})


# ── dialogue ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("val,exp", [(0, False), (1, True), (2, True)])
def test_is_text_box_open(val, exp):
    assert is_text_box_open({"dialogue": val}) == exp


# ── party queries ────────────────────────────────────────────────────────────

def test_get_lead_full_hp():
    state = {"party": _make_party((20, 20))}
    assert get_lead_hp_ratio(state) == 1.0


def test_get_lead_half_hp():
    state = {"party": _make_party((10, 20))}
    assert get_lead_hp_ratio(state) == pytest.approx(0.5)


def test_get_lead_empty_party():
    assert get_lead_hp_ratio({"party": []}) == 1.0


def test_count_alive_pokemon():
    state = {"party": _make_party((10, 20), (0, 15), (5, 5))}
    assert count_alive_pokemon(state) == 2


def test_count_badges():
    assert count_badges({"badges": 0b00000101}) == 2
    assert count_badges({"badges": 0}) == 0
    assert count_badges({"badges": 0xFF}) == 8


# ── enemy queries ────────────────────────────────────────────────────────────

def test_get_enemy_hp_ratio_in_battle():
    state = {"enemy": _make_enemy(hp=15, max_hp=30)}
    assert get_enemy_hp_ratio(state) == pytest.approx(0.5)


def test_get_enemy_hp_ratio_no_battle():
    assert get_enemy_hp_ratio({"enemy": None}) == 1.0


def test_get_enemy_stats_no_battle():
    stats = get_enemy_stats({"enemy": None})
    assert stats == {"atk_stat": 0, "def_stat": 0, "spd_stat": 0}


def test_get_enemy_stats_in_battle():
    state = {"enemy": _make_enemy(atk_stat=25, def_stat=18, spd_stat=22)}
    stats = get_enemy_stats(state)
    assert stats["atk_stat"] == 25


# ── stat stages ──────────────────────────────────────────────────────────────

def test_player_stages_default_to_neutral():
    stages = get_player_stages({})
    assert all(v == NEUTRAL_STAGE for v in stages.values())


def test_enemy_stages_default_to_neutral():
    stages = get_enemy_stages({})
    assert all(v == NEUTRAL_STAGE for v in stages.values())


def test_enemy_stat_lowered_detects_debuff():
    prev = _neutral_stages()
    cur  = {**prev, "atk": NEUTRAL_STAGE - 1}
    assert enemy_stat_lowered(cur, prev)


def test_enemy_stat_lowered_no_change():
    stages = _neutral_stages()
    assert not enemy_stat_lowered(stages, stages)


def test_enemy_stat_lowered_buff_not_debuff():
    prev = _neutral_stages()
    cur  = {**prev, "atk": NEUTRAL_STAGE + 1}   # enemy was buffed, not lowered
    assert not enemy_stat_lowered(cur, prev)
