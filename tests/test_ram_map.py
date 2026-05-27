"""
Unit tests for ram_map.py — stage multipliers, catch probability, status helpers.
No ROM or emulator needed.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
from ram_map import (
    NEUTRAL_STAGE, stage_mult,
    catch_probability,
    status_mult, status_tag,
    STATUS_SLP_MASK, STATUS_PAR_MASK, STATUS_FRZ_MASK,
    POKECENTER_MAPS,
    MAP_SIZE, MAP_HEIGHT,
)


# ── Stage multipliers ────────────────────────────────────────────────────────

def test_neutral_stage_is_1x():
    """Stage 7 (NEUTRAL_STAGE) must give a multiplier of 1.0."""
    assert stage_mult(NEUTRAL_STAGE) == pytest.approx(1.0)


def test_stage_above_neutral_increases():
    """Each stage above neutral should give a multiplier > 1."""
    assert stage_mult(NEUTRAL_STAGE + 1) > 1.0
    assert stage_mult(NEUTRAL_STAGE + 2) > stage_mult(NEUTRAL_STAGE + 1)


def test_stage_below_neutral_decreases():
    """Each stage below neutral should give a multiplier < 1."""
    assert stage_mult(NEUTRAL_STAGE - 1) < 1.0
    assert stage_mult(NEUTRAL_STAGE - 2) < stage_mult(NEUTRAL_STAGE - 1)


def test_stage_clamps_out_of_range():
    """stage_mult should clamp to neutral and not crash for out-of-range values."""
    low  = stage_mult(0)    # clamps to neutral
    high = stage_mult(13)   # max valid
    assert low  == pytest.approx(1.0)   # clamped to neutral
    assert high > 1


# ── Status helpers ───────────────────────────────────────────────────────────

def test_no_status_is_healthy():
    assert status_mult(0) == pytest.approx(1.0)


def test_sleep_reduces_multiplier():
    slp_byte = 0b00000011  # bits 0-2: SLP counter
    assert status_mult(slp_byte) < 1.0


def test_paralysis_reduces_multiplier():
    assert status_mult(STATUS_PAR_MASK) < 1.0


def test_freeze_reduces_multiplier():
    assert status_mult(STATUS_FRZ_MASK) < 1.0


def test_status_tag_healthy_is_empty():
    assert status_tag(0) == ""


def test_status_tag_paralysis():
    assert status_tag(STATUS_PAR_MASK) == "[PAR]"


def test_status_tag_sleep():
    assert status_tag(STATUS_SLP_MASK) == "[SLP]"


# ── Catch probability ────────────────────────────────────────────────────────

def test_catch_prob_fainted_returns_0():
    """max_hp=0 → division guard → 0.0."""
    p = catch_probability(catch_rate=45, hp=0, max_hp=0, status=0)
    assert p == 0.0


def test_catch_prob_in_range():
    p = catch_probability(catch_rate=45, hp=5, max_hp=30, status=0)
    assert 0.0 <= p <= 1.0


def test_catch_prob_low_hp_higher_than_full():
    """Lower HP should give equal or higher catch probability."""
    p_low  = catch_probability(catch_rate=45, hp=1,  max_hp=30, status=0)
    p_full = catch_probability(catch_rate=45, hp=30, max_hp=30, status=0)
    assert p_low >= p_full


def test_catch_prob_sleep_better_than_healthy():
    """SLP status multiplier (×2) should improve catch probability."""
    p_ok  = catch_probability(catch_rate=45, hp=10, max_hp=30, status=0)
    p_slp = catch_probability(catch_rate=45, hp=10, max_hp=30,
                              status=STATUS_SLP_MASK)
    assert p_slp >= p_ok


def test_catch_prob_guaranteed_for_high_catch_rate():
    """Catch rate 255 at 1 HP gives very high probability (close to 1.0)."""
    p = catch_probability(catch_rate=255, hp=1, max_hp=30, status=0)
    assert p > 0.9  # Should be close to guaranteed


# ── Map constants integrity ──────────────────────────────────────────────────

def test_pokecenter_maps_nonempty():
    assert len(POKECENTER_MAPS) > 0


def test_map_size_and_height_keys_consistent():
    """Every key in MAP_SIZE should also appear in MAP_HEIGHT."""
    extra = set(MAP_SIZE) - set(MAP_HEIGHT)
    assert not extra, f"MAP_SIZE keys missing from MAP_HEIGHT: {extra}"


def test_map_dimensions_positive():
    for k in MAP_SIZE:
        assert MAP_SIZE[k] > 0
        assert MAP_HEIGHT[k] > 0
