"""
game_helpers.py — RAM reading and pure game-state accessors.

Two layers:
  1. READ layer  — functions that take a PyBoy memory object and return data.
                   read_state(memory) is the single entry point used by the env.
  2. QUERY layer — pure functions that take the state dict returned by read_state()
                   and extract / derive values for reward / observation logic.
"""

from __future__ import annotations

from ram_map import (
    POKECENTER_MAPS, NEUTRAL_STAGE,
    ADDR_MAP_ID, ADDR_PLAYER_X, ADDR_PLAYER_Y, ADDR_IN_BATTLE, ADDR_BADGES,
    ADDR_TEXT_BOX, ADDR_NPC_TALK_FLAG, ADDR_NUM_BAG_ITEMS,
    ADDR_PARTY_COUNT, ADDR_PARTY_DATA, ADDR_PARTY_SPECIES,
    ADDR_ENEMY_HP, ADDR_ENEMY_MAX_HP, ADDR_ENEMY_SPECIES, ADDR_ENEMY_LEVEL,
    ADDR_ENEMY_STATUS, ADDR_ENEMY_ATK, ADDR_ENEMY_DEF, ADDR_ENEMY_SPD,
    ADDR_ENEMY_CATCH_RATE,
    ADDR_PLAYER_ATK_STAGE, ADDR_PLAYER_DEF_STAGE, ADDR_PLAYER_SPD_STAGE,
    ADDR_PLAYER_SPC_STAGE, ADDR_PLAYER_ACC_STAGE, ADDR_PLAYER_EVA_STAGE,
    ADDR_ENEMY_ATK_STAGE, ADDR_ENEMY_DEF_STAGE, ADDR_ENEMY_SPD_STAGE,
    ADDR_ENEMY_SPC_STAGE, ADDR_ENEMY_ACC_STAGE, ADDR_ENEMY_EVA_STAGE,
    ADDR_POKEDEX_OWNED, ADDR_BAG_ITEMS,
    PARTY_MON_SIZE,
    OFF_HP_HI, OFF_HP_LO, OFF_MAX_HP_HI, OFF_MAX_HP_LO,
    OFF_LEVEL, OFF_STATUS,
    OFF_EXP_HI, OFF_EXP_MID, OFF_EXP_LO,
    OFF_MOVE_0, OFF_MOVE_1, OFF_MOVE_2, OFF_MOVE_3,
    OFF_PP_0, OFF_PP_1, OFF_PP_2, OFF_PP_3,
    OFF_ATK_HI, OFF_ATK_LO, OFF_DEF_HI, OFF_DEF_LO, OFF_SPD_HI, OFF_SPD_LO,
    BALL_ITEM_IDS, HEAL_ITEM_IDS, HM_ITEM_IDS,
)

_STAT_KEYS: tuple[str, ...] = ("atk", "def", "spd", "spc", "acc", "eva")
_NEUTRAL_STAGES: dict[str, int] = {k: NEUTRAL_STAGE for k in _STAT_KEYS}


# ═════════════════════════════════════════════════════════════════════════════
# READ LAYER — take a PyBoy memory object, return structured data
# ═════════════════════════════════════════════════════════════════════════════

def read_party(memory) -> list[dict]:
    """Read up to 6 party Pokémon from RAM."""
    count = memory[ADDR_PARTY_COUNT]
    party = []
    for i in range(min(count, 6)):
        base   = ADDR_PARTY_DATA + i * PARTY_MON_SIZE
        hp     = (memory[base + OFF_HP_HI] << 8) | memory[base + OFF_HP_LO]
        max_hp = (memory[base + OFF_MAX_HP_HI] << 8) | memory[base + OFF_MAX_HP_LO]
        level  = memory[base + OFF_LEVEL]
        status = memory[base + OFF_STATUS]
        exp    = (memory[base + OFF_EXP_HI] << 16) | (memory[base + OFF_EXP_MID] << 8) | memory[base + OFF_EXP_LO]
        moves  = [memory[base + OFF_MOVE_0], memory[base + OFF_MOVE_1],
                  memory[base + OFF_MOVE_2], memory[base + OFF_MOVE_3]]
        pps    = [memory[base + OFF_PP_0], memory[base + OFF_PP_1],
                  memory[base + OFF_PP_2], memory[base + OFF_PP_3]]
        party.append({
            "hp":       hp,
            "max_hp":   max(max_hp, 1),
            "level":    max(level, 1),
            "status":   status,
            "exp":      exp,
            "moves":    moves,
            "pp":       pps,
            "species":  memory[ADDR_PARTY_SPECIES + i],
            "atk_stat": (memory[base + OFF_ATK_HI] << 8) | memory[base + OFF_ATK_LO],
            "def_stat": (memory[base + OFF_DEF_HI] << 8) | memory[base + OFF_DEF_LO],
            "spd_stat": (memory[base + OFF_SPD_HI] << 8) | memory[base + OFF_SPD_LO],
        })
    return party


def read_battle_enemy(memory) -> dict | None:
    """Read enemy battle data. Returns None when not in battle."""
    if memory[ADDR_IN_BATTLE] == 0:
        return None
    hp     = (memory[ADDR_ENEMY_HP]     << 8) | memory[ADDR_ENEMY_HP + 1]
    max_hp = (memory[ADDR_ENEMY_MAX_HP] << 8) | memory[ADDR_ENEMY_MAX_HP + 1]
    # Guard against uninitialised RAM at the first frames of battle
    max_hp = max(max_hp, hp, 1)
    return {
        "hp":         hp,
        "max_hp":     max(max_hp, 1),
        "species":    memory[ADDR_ENEMY_SPECIES],
        "level":      memory[ADDR_ENEMY_LEVEL],
        "status":     memory[ADDR_ENEMY_STATUS],
        "atk_stat":   (memory[ADDR_ENEMY_ATK] << 8) | memory[ADDR_ENEMY_ATK + 1],
        "def_stat":   (memory[ADDR_ENEMY_DEF] << 8) | memory[ADDR_ENEMY_DEF + 1],
        "spd_stat":   (memory[ADDR_ENEMY_SPD] << 8) | memory[ADDR_ENEMY_SPD + 1],
        "catch_rate": memory[ADDR_ENEMY_CATCH_RATE],
    }


def read_battle_stat_stages(memory) -> dict:
    """
    Read in-battle stat stages for player and enemy.
    Each sub-dict: atk, def, spd, spc, acc, eva — raw stage (0-13, neutral=7).
    Returns all-neutral dicts when not in battle.
    """
    neutral = _NEUTRAL_STAGES.copy()
    if memory[ADDR_IN_BATTLE] == 0:
        return {"player": neutral.copy(), "enemy": neutral.copy()}
    return {
        "player": {
            "atk": memory[ADDR_PLAYER_ATK_STAGE],
            "def": memory[ADDR_PLAYER_DEF_STAGE],
            "spd": memory[ADDR_PLAYER_SPD_STAGE],
            "spc": memory[ADDR_PLAYER_SPC_STAGE],
            "acc": memory[ADDR_PLAYER_ACC_STAGE],
            "eva": memory[ADDR_PLAYER_EVA_STAGE],
        },
        "enemy": {
            "atk": memory[ADDR_ENEMY_ATK_STAGE],
            "def": memory[ADDR_ENEMY_DEF_STAGE],
            "spd": memory[ADDR_ENEMY_SPD_STAGE],
            "spc": memory[ADDR_ENEMY_SPC_STAGE],
            "acc": memory[ADDR_ENEMY_ACC_STAGE],
            "eva": memory[ADDR_ENEMY_EVA_STAGE],
        },
    }


def read_pokedex_owned(memory) -> frozenset[int]:
    """Read the 19-byte Pokédex 'owned' bitfield (1 bit per species, 1-151)."""
    owned = set()
    for byte_i in range(19):
        byte = memory[ADDR_POKEDEX_OWNED + byte_i]
        for bit in range(8):
            species = byte_i * 8 + bit + 1
            if species <= 151 and (byte >> bit) & 1:
                owned.add(species)
    return frozenset(owned)


def read_bag_items(memory) -> dict:
    """Return a simplified bag summary: {'balls': N, 'heals': N, 'hm_cut': bool, ...}."""
    n_slots = memory[ADDR_NUM_BAG_ITEMS]
    balls = heals = 0
    hm_flags: dict[str, bool] = {k: False for k in HM_ITEM_IDS}
    for i in range(min(n_slots, 20)):
        item_id = memory[ADDR_BAG_ITEMS + i * 2]
        count   = memory[ADDR_BAG_ITEMS + i * 2 + 1]
        if item_id in BALL_ITEM_IDS:
            balls += count
        if item_id in HEAL_ITEM_IDS:
            heals += count
        for name, hm_id in HM_ITEM_IDS.items():
            if item_id == hm_id:
                hm_flags[name] = True
    return {"balls": balls, "heals": heals, **hm_flags}


def read_state(memory) -> dict:
    """
    Single entry point: read all game state from RAM into a plain dict.
    This is the ONLY place in the codebase that touches PyBoy memory directly.
    """
    return {
        "map_id":        memory[ADDR_MAP_ID],
        "player_x":      memory[ADDR_PLAYER_X],
        "player_y":      memory[ADDR_PLAYER_Y],
        "in_battle":     memory[ADDR_IN_BATTLE],       # 0=none, 1=wild, 2=trainer
        "badges":        memory[ADDR_BADGES],
        "party":         read_party(memory),
        "enemy":         read_battle_enemy(memory),
        "items":         read_bag_items(memory),
        "pokedex_owned": read_pokedex_owned(memory),
        "dialogue":      1 if memory[ADDR_TEXT_BOX] != 0 else 0,
        "stages":        read_battle_stat_stages(memory),
        "bag_count":     memory[ADDR_NUM_BAG_ITEMS],   # raw slot count for item-pickup detection
        "npc_talk_flag": memory[ADDR_NPC_TALK_FLAG],   # 0=none, 2=NPC dialogue (not start menu)
    }


# ═════════════════════════════════════════════════════════════════════════════
# QUERY LAYER — pure functions over the state dict
# ═════════════════════════════════════════════════════════════════════════════

# ── Map / position ────────────────────────────────────────────────────────────

def get_current_map(state: dict) -> int:
    """Return the current map ID."""
    return state["map_id"]


def get_player_pos(state: dict) -> tuple[int, int]:
    """Return (x, y) player tile position."""
    return state["player_x"], state["player_y"]


def is_in_pokecenter(state: dict) -> bool:
    """True when the player is inside any Pokémon Center."""
    return state["map_id"] in POKECENTER_MAPS


# ── Battle state ──────────────────────────────────────────────────────────────

def is_in_battle(state: dict) -> bool:
    """True when any battle (wild or trainer) is active."""
    return state.get("in_battle", 0) > 0


def is_trainer_battle(state: dict) -> bool:
    """True when the current battle is against a trainer (in_battle == 2)."""
    return state.get("in_battle", 0) == 2


def is_wild_battle(state: dict) -> bool:
    """True when the current battle is a wild encounter (in_battle == 1)."""
    return state.get("in_battle", 0) == 1


# ── UI / dialogue ─────────────────────────────────────────────────────────────

def is_text_box_open(state: dict) -> bool:
    """True when any text box / dialogue is displayed (ADDR_TEXT_BOX != 0)."""
    return bool(state.get("dialogue", 0))


# ── Party ─────────────────────────────────────────────────────────────────────

def get_lead(state: dict) -> dict | None:
    """Return the first (lead) party Pokémon dict, or None if party is empty."""
    party = state.get("party", [])
    return party[0] if party else None


def get_lead_hp_ratio(state: dict) -> float:
    """Return lead Pokémon HP / max_HP in [0.0, 1.0]. Returns 1.0 if no party."""
    lead = get_lead(state)
    if lead is None:
        return 1.0
    return lead["hp"] / max(lead["max_hp"], 1)


def count_alive_pokemon(state: dict) -> int:
    """Count party members with HP > 0."""
    return sum(1 for p in state.get("party", []) if p.get("hp", 0) > 0)


def count_badges(state: dict) -> int:
    """Return the number of gym badges earned (popcount of badge byte)."""
    return bin(state.get("badges", 0)).count("1")


# ── Enemy ─────────────────────────────────────────────────────────────────────

def get_enemy(state: dict) -> dict | None:
    """Return the enemy battle data dict, or None when not in battle."""
    return state.get("enemy")


def get_enemy_hp(state: dict) -> int:
    """Return current enemy HP, or 0 if not in battle."""
    enemy = get_enemy(state)
    return enemy["hp"] if enemy else 0


def get_enemy_hp_ratio(state: dict) -> float:
    """Return enemy HP / max_HP in [0.0, 1.0]. Returns 1.0 if not in battle."""
    enemy = get_enemy(state)
    if enemy is None:
        return 1.0
    return enemy["hp"] / max(enemy["max_hp"], 1)


def get_enemy_stats(state: dict) -> dict[str, int]:
    """
    Return enemy combat stats as {'atk_stat', 'def_stat', 'spd_stat'}.
    Returns zeros when not in battle.
    """
    enemy = get_enemy(state)
    if enemy is None:
        return {"atk_stat": 0, "def_stat": 0, "spd_stat": 0}
    return {k: enemy[k] for k in ("atk_stat", "def_stat", "spd_stat")}


# ── Stat stages ───────────────────────────────────────────────────────────────

def get_enemy_stages(state: dict) -> dict[str, int]:
    """
    Return enemy in-battle stat stages for atk/def/spd/spc/acc/eva.
    All values default to NEUTRAL_STAGE (7) when not in battle.
    """
    return state.get("stages", {}).get("enemy", _NEUTRAL_STAGES.copy())


def get_player_stages(state: dict) -> dict[str, int]:
    """
    Return player in-battle stat stages for atk/def/spd/spc/acc/eva.
    All values default to NEUTRAL_STAGE (7) when not in battle.
    """
    return state.get("stages", {}).get("player", _NEUTRAL_STAGES.copy())


def enemy_stat_lowered(cur_stages: dict[str, int], prev_stages: dict[str, int]) -> bool:
    """
    Return True if any enemy stat stage decreased between two consecutive steps.
    Covers debuff moves: Growl, Leer, Sand Attack, String Shot, etc.
    """
    return any(
        cur_stages.get(k, NEUTRAL_STAGE) < prev_stages.get(k, NEUTRAL_STAGE)
        for k in _STAT_KEYS
    )
