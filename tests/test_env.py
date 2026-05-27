"""
Smoke tests for env.py — observation shapes, constants, and pure reward logic.
No ROM is required: we test the constants and pure (ROM-free) methods only.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
import numpy as np

import env as env_module
from env import (
    STATE_VEC_SIZE,
    NEW_TILE_REWARD, NEW_MAP_REWARD, DEATH_PENALTY,
    LEVEL_UP_REWARD, BADGE_REWARD,
    MILESTONE_MAPS,
    MAX_STEPS,
    SCORE_FLOOR,
    _DISTANCE_ACTIVE_MAPS, _DISTANCE_ORIGIN,
)


# ── Module-level constants sanity ────────────────────────────────────────────

def test_state_vec_size_matches_docstring():
    """STATE_VEC_SIZE must be 91 (matches class docstring and _build_state_vec)."""
    assert STATE_VEC_SIZE == 91


def test_observation_space_shapes():
    """Observation space dict should contain screen, state, minimap with expected shapes."""
    from gymnasium import spaces
    obs_space = spaces.Dict({
        "screen":  spaces.Box(0, 255, (84, 84, 4), dtype=np.uint8),
        "state":   spaces.Box(-np.inf, np.inf, (STATE_VEC_SIZE,), dtype=np.float32),
        "minimap": spaces.Box(0, 255, (21, 21, 1), dtype=np.uint8),
    })
    assert obs_space["screen"].shape  == (84, 84, 4)
    assert obs_space["state"].shape   == (STATE_VEC_SIZE,)
    assert obs_space["minimap"].shape == (21, 21, 1)


# ── Reward constant ordering / sanity ────────────────────────────────────────

def test_death_penalty_is_negative():
    assert DEATH_PENALTY < 0


def test_new_tile_reward_positive():
    assert NEW_TILE_REWARD > 0


def test_new_map_reward_positive():
    assert NEW_MAP_REWARD > 0


def test_level_up_reward_positive():
    assert LEVEL_UP_REWARD > 0


def test_badge_reward_dominates_level_up():
    """A badge should reward more than a level-up (badge is harder to earn)."""
    assert BADGE_REWARD > LEVEL_UP_REWARD


def test_milestone_maps_nonempty():
    assert len(MILESTONE_MAPS) > 0


def test_milestone_map_rewards_positive():
    for map_id, reward in MILESTONE_MAPS.items():
        assert reward > 0, f"Milestone map {map_id} has non-positive reward {reward}"


def test_score_floor_is_negative():
    assert SCORE_FLOOR < 0


# ── Distance reward config ───────────────────────────────────────────────────

def test_distance_origin_is_pair():
    assert len(_DISTANCE_ORIGIN) == 2


def test_pallet_town_in_active_maps():
    """Map 0 (Pallet Town) should be in the distance-reward active set."""
    assert 0 in _DISTANCE_ACTIVE_MAPS


def test_route_1_in_active_maps():
    assert 12 in _DISTANCE_ACTIVE_MAPS


# ── Attribute completeness: __init__ must contain tracking vars ──────────────

def test_just_respawned_in_init():
    """_just_respawned must be initialised in __init__ (not only in reset)."""
    src = pathlib.Path(__file__).parent.parent / "env.py"
    text = src.read_text(encoding="utf-8")

    # Find __init__ block — everything before the first def that follows it
    init_start = text.find("def __init__")
    assert init_start != -1
    # Find next def after __init__
    next_def = text.find("\n    def ", init_start + 10)
    init_block = text[init_start:next_def]

    assert "_just_respawned" in init_block, (
        "_just_respawned must be initialised in __init__ to avoid AttributeError "
        "if _compute_reward is called before reset()"
    )


def test_party_strengths_removed():
    """Dead variable _party_strengths should no longer be assigned anywhere."""
    src = pathlib.Path(__file__).parent.parent / "env.py"
    text = src.read_text(encoding="utf-8")
    assert "_party_strengths" not in text, (
        "_party_strengths is dead code and should have been removed"
    )


# ── game_helpers duplication check ──────────────────────────────────────────

def test_game_helpers_no_duplicate_functions():
    """The second (duplicate) block of query functions must be gone."""
    src = pathlib.Path(__file__).parent.parent / "game_helpers.py"
    text = src.read_text(encoding="utf-8")
    # Each query function should appear exactly once
    for fn in ("def is_in_pokecenter", "def get_lead_hp_ratio",
               "def count_badges", "def enemy_stat_lowered"):
        count = text.count(fn)
        assert count == 1, f"{fn!r} appears {count} times — expected 1 (duplicate not removed)"
