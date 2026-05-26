"""
ram_map.py — Single source of truth for Pokemon Red RAM addresses,
map data, stage tables, item IDs, and party struct offsets.

All other modules (env.py, setup_state.py, play.py, multiplay.py, …)
must import from here. Never define addresses locally in those files.

Source: https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map
        (verified 2026-05-24 against Pokemon Red English/INT ROM)
"""

from __future__ import annotations

# ── Core RAM addresses ────────────────────────────────────────────────────────
ADDR_PLAYER_Y       = 0xD361   # current player Y tile position
ADDR_PLAYER_X       = 0xD362   # current player X tile position
ADDR_MAP_ID         = 0xD35E   # current map number
ADDR_IN_BATTLE      = 0xD057   # battle type: 0=none, 1=wild, 2=trainer
ADDR_PARTY_COUNT    = 0xD163   # number of Pokemon in party
ADDR_PARTY_SPECIES  = 0xD164   # 6-byte species array (+ 0xFF terminator)
ADDR_PARTY_DATA     = 0xD16B   # start of party struct array
ADDR_BADGES         = 0xD356   # badge bitfield (bit0=Boulder … bit7=Earth)
ADDR_NUM_BAG_ITEMS  = 0xD31D   # number of bag item slots in use
ADDR_BAG_ITEMS      = 0xD31E   # pairs of (item_id, quantity)
ADDR_POKEDEX_OWNED  = 0xD2F7   # 19 bytes, 1 bit per species, owned flags
ADDR_POKEDEX_SEEN   = 0xD30A   # 19 bytes, 1 bit per species, seen flags
ADDR_MONEY_1        = 0xD347   # money byte 1 (3-byte BCD: D347-D349)
ADDR_PLAYER_NAME    = 0xD158   # player name (10 bytes)
ADDR_RIVAL_NAME     = 0xD34A   # rival name (8 bytes)
ADDR_GAME_HOURS     = 0xDA40   # play time hours (2 bytes)
ADDR_GAME_MINUTES   = 0xDA42   # play time minutes (2 bytes)
ADDR_GAME_SECONDS   = 0xDA44   # play time seconds (1 byte)
ADDR_OPTIONS        = 0xD355   # options: bit7=noAnim, bit6=set, low4=textSpeed
ADDR_TILESET_TYPE   = 0xFFD7   # tileset: 0=indoors, 1=cave, 2=outside
ADDR_AUDIO_TRACK    = 0xD35B   # current map audio track
ADDR_AUDIO_BANK     = 0xD35C   # current map audio bank
ADDR_TEXT_BOX       = 0xCFC4   # 0=no UI, 1=any text box/menu open (verified via memory diff)
ADDR_NPC_TALK_FLAG  = 0xCF13   # 0→2 when talking to NPC specifically (not menu, verified via diff)
ADDR_JOYPAD_HELD    = 0xFFB4   # held buttons this frame
ADDR_BATTLE_TURN    = 0xFFF3   # battle turn: 0=player, 1=opponent

# ── Enemy in-battle struct (all INT/English addresses) ────────────────────────
ADDR_ENEMY_SPECIES  = 0xCFD8   # enemy internal species ID
ADDR_ENEMY_HP       = 0xCFE6   # enemy current HP (2 bytes big-endian, CFE6-CFE7)
ADDR_ENEMY_LEVEL_Q  = 0xCFE8   # enemy level (quick copy)
ADDR_ENEMY_STATUS   = 0xCFE9   # enemy status byte
ADDR_ENEMY_TYPE1    = 0xCFEA   # enemy type 1
ADDR_ENEMY_TYPE2    = 0xCFEB   # enemy type 2
ADDR_ENEMY_LEVEL    = 0xCFF3   # enemy level (stat-calculated copy)
ADDR_ENEMY_MAX_HP   = 0xCFF4   # enemy max HP (2 bytes big-endian, CFF4-CFF5)
ADDR_ENEMY_ATK      = 0xCFF6   # enemy attack stat (2 bytes)
ADDR_ENEMY_DEF      = 0xCFF8   # enemy defense stat (2 bytes)
ADDR_ENEMY_SPD      = 0xCFFA   # enemy speed stat (2 bytes)
ADDR_ENEMY_SPC      = 0xCFFC   # enemy special stat (2 bytes)
ADDR_ENEMY_CATCH_RATE = 0xCFEC # enemy catch rate (1 byte, from species base stats)

# ── Player in-battle slot ─────────────────────────────────────────────────────
ADDR_PLAYER_BATTLE_SPECIES = 0xD014  # player's active mon species
ADDR_PLAYER_BATTLE_HP      = 0xD015  # player's active mon HP (2 bytes)
ADDR_PLAYER_BATTLE_STATUS  = 0xD018  # player's active mon status
ADDR_PLAYER_BATTLE_MOVE1   = 0xD01C  # moves 1-4 (D01C-D01F)
ADDR_PLAYER_BATTLE_LEVEL   = 0xD022  # player's active mon level
ADDR_PLAYER_BATTLE_MAX_HP  = 0xD023  # player's active mon max HP (2 bytes)

# ── In-battle stat stage modifiers (neutral = 7, valid range 1-13) ────────────
# Player stages: CD1A-CD1F  |  Enemy stages: CD2E-CD33
ADDR_PLAYER_ATK_STAGE = 0xCD1A
ADDR_PLAYER_DEF_STAGE = 0xCD1B
ADDR_PLAYER_SPD_STAGE = 0xCD1C
ADDR_PLAYER_SPC_STAGE = 0xCD1D
ADDR_PLAYER_ACC_STAGE = 0xCD1E
ADDR_PLAYER_EVA_STAGE = 0xCD1F
ADDR_ENEMY_ATK_STAGE  = 0xCD2E
ADDR_ENEMY_DEF_STAGE  = 0xCD2F
ADDR_ENEMY_SPD_STAGE  = 0xCD30
ADDR_ENEMY_SPC_STAGE  = 0xCD31
ADDR_ENEMY_ACC_STAGE  = 0xCD32
ADDR_ENEMY_EVA_STAGE  = 0xCD33

# ── Battle misc ───────────────────────────────────────────────────────────────
ADDR_BATTLE_TURNS       = 0xCCD5   # number of turns in current battle
ADDR_PLAYER_MOVE_USED   = 0xCCDC   # player-selected move this turn
ADDR_ENEMY_MOVE_USED    = 0xCCDD   # enemy-selected move this turn
ADDR_CRIT_FLAG          = 0xD05E   # 01=crit hit, 02=OHKO
ADDR_DAMAGE_AMOUNT      = 0xD0D8   # damage about to be dealt
ADDR_BATTLE_STATUS_P1   = 0xD062   # player battle status byte 1 (confusion etc.)
ADDR_BATTLE_STATUS_P2   = 0xD063   # player battle status byte 2 (substitute etc.)
ADDR_BATTLE_STATUS_E1   = 0xD067   # enemy battle status byte 1
ADDR_BATTLE_STATUS_E2   = 0xD068   # enemy battle status byte 2

# ── Wild encounter data ───────────────────────────────────────────────────────
ADDR_ENCOUNTER_RATES    = 0xD887   # encounter rate for current area
ADDR_ENCOUNTER_DATA     = 0xD888   # level+species pairs (4 common, 4 uncommon, 2 rare)

# ── Party struct layout (44-byte struct per slot) ─────────────────────────────
PARTY_MON_SIZE = 44   # bytes per party slot
OFF_HP_HI      = 0x01   # current HP high byte
OFF_HP_LO      = 0x02   # current HP low byte
OFF_STATUS     = 0x04   # status condition byte
OFF_MOVE_0     = 0x08   # move 1 ID
OFF_MOVE_1     = 0x09
OFF_MOVE_2     = 0x0A
OFF_MOVE_3     = 0x0B
OFF_EXP_HI     = 0x0E   # experience (3 bytes big-endian)
OFF_EXP_MID    = 0x0F
OFF_EXP_LO     = 0x10
OFF_PP_0       = 0x1D   # PP for moves 1-4
OFF_PP_1       = 0x1E
OFF_PP_2       = 0x1F
OFF_PP_3       = 0x20
OFF_LEVEL      = 0x21   # actual level (not the false 'level' at +3)
OFF_MAX_HP_HI  = 0x22
OFF_MAX_HP_LO  = 0x23
OFF_ATK_HI     = 0x24
OFF_ATK_LO     = 0x25
OFF_DEF_HI     = 0x26
OFF_DEF_LO     = 0x27
OFF_SPD_HI     = 0x28
OFF_SPD_LO     = 0x29
OFF_SPC_HI     = 0x2A
OFF_SPC_LO     = 0x2B

# ── Status byte masks ─────────────────────────────────────────────────────────
STATUS_SLP_MASK = 0x07   # bits 0-2: sleep counter (> 0 = asleep)
STATUS_PSN_MASK = 0x08   # bit 3
STATUS_BRN_MASK = 0x10   # bit 4
STATUS_FRZ_MASK = 0x20   # bit 5
STATUS_PAR_MASK = 0x40   # bit 6

STATUS_NAMES: dict[int, str] = {
    STATUS_SLP_MASK: "SLP",
    STATUS_FRZ_MASK: "FRZ",
    STATUS_PAR_MASK: "PAR",
    STATUS_BRN_MASK: "BRN",
    STATUS_PSN_MASK: "PSN",
}

# ── Gen-1 stat-stage multiplier lookup ───────────────────────────────────────
# Stored value range: 1-13.  Neutral = 7.  Index = stored_value - 1.
# Formula: max(2, 2+s) / max(2, 2-s)  where s = stored_value - 7.
STAGE_MULT: tuple[float, ...] = (
    2/8, 2/7, 2/6, 2/5, 2/4, 2/3,   # stages -6 … -1  (stored 1 … 6)
    1.0,                              # stage   0        (stored 7)
    3/2, 4/2, 5/2, 6/2, 7/2, 8/2,   # stages +1 … +6  (stored 8 … 13)
)
NEUTRAL_STAGE: int = 7


def stage_mult(raw: int) -> float:
    """Return the multiplier for a raw stage byte. Out-of-range → neutral."""
    if not (1 <= raw <= 13):
        raw = NEUTRAL_STAGE
    return STAGE_MULT[raw - 1]


def fmt_stage(raw: int) -> str:
    """Format a raw stage byte as '+N' / '-N' / '±0'."""
    if not (1 <= raw <= 13):
        raw = NEUTRAL_STAGE
    s = raw - NEUTRAL_STAGE
    return f"+{s}" if s > 0 else ("±0" if s == 0 else str(s))


def status_tag(s: int) -> str:
    """Return a short status tag string like '[PAR]', or '' if healthy."""
    for mask, name in STATUS_NAMES.items():
        if s & mask:
            return f"[{name}]"
    return ""


def status_mult(s: int) -> float:
    """Convert a status byte into a combat effectiveness multiplier (legacy)."""
    if s & STATUS_SLP_MASK: return 0.3
    if s & STATUS_FRZ_MASK: return 0.2
    if s & STATUS_PAR_MASK: return 0.6
    if s & STATUS_BRN_MASK: return 0.7
    if s & STATUS_PSN_MASK: return 0.85
    return 1.0


def status_offense_mult(s: int) -> float:
    """
    Expected fraction of turns the Pokémon can actually attack (Gen 1).
      SLP / FRZ : 0.0  — can't move at all
      PAR       : 0.75 — 25% full-paralysis each turn
      BRN       : 0.5  — burn halves ATK in the damage formula
      PSN       : 1.0  — poison doesn't reduce offence
    """
    if s & STATUS_SLP_MASK: return 0.0
    if s & STATUS_FRZ_MASK: return 0.0
    if s & STATUS_PAR_MASK: return 0.75
    if s & STATUS_BRN_MASK: return 0.5
    return 1.0


def status_passive_dmg(s: int, max_hp: int) -> float:
    """
    HP lost per turn from status damage alone (PSN or BRN = 1/16 of max HP).
    This is independent of the opponent's attacks.
    """
    if s & (STATUS_PSN_MASK | STATUS_BRN_MASK):
        return max_hp / 16.0
    return 0.0


def catch_probability(catch_rate: int, hp: int, max_hp: int, status: int) -> float:
    """
    Simplified Gen 1 catch probability in [0, 1] for a standard Poké Ball.

    Formula (Gen 1):  a = (3*max_hp - 2*hp) * catch_rate * status_mult / (3*max_hp)
    If a >= 255 → guaranteed catch (return 1.0).
    Otherwise the game does 4 random checks each needing rand(0,255) < a;
    P(all 4 pass) = (a/255)^4  — returned here.

    Status catch multipliers: SLP/FRZ = ×2.0, BRN/PSN/PAR = ×1.5, none = ×1.0
    """
    SLP_MASK = 0b00000111
    FRZ_MASK = 0b00100000
    BRN_MASK = 0b00010000
    PSN_MASK = 0b00001000
    PAR_MASK = 0b01000000
    if   status & (SLP_MASK | FRZ_MASK): s_mult = 2.0
    elif status & (BRN_MASK | PSN_MASK | PAR_MASK): s_mult = 1.5
    else: s_mult = 1.0

    if max_hp <= 0:
        return 0.0
    a = (3 * max_hp - 2 * hp) * catch_rate * s_mult / (3 * max_hp)
    if a >= 255:
        return 1.0
    return (a / 255.0) ** 4


# ── Item IDs ──────────────────────────────────────────────────────────────────
BALL_ITEM_IDS: frozenset[int] = frozenset({0x01, 0x02, 0x03, 0x04})
# 0x01=Master Ball, 0x02=Ultra Ball, 0x03=Great Ball, 0x04=Poke Ball

HEAL_ITEM_IDS: frozenset[int] = frozenset({0x14, 0x15, 0x16, 0x17, 0x18, 0x19})
# 0x14=Potion, 0x15=Super Potion, 0x16=Hyper Potion, 0x17=Max Potion

HM_ITEM_IDS: dict[str, int] = {
    "cut":      0xC4,   # HM01
    "fly":      0xC5,   # HM02
    "surf":     0xC6,   # HM03
    "strength": 0xC7,   # HM04
}

# ── Audio track IDs (Bank 08) — useful for detecting game state ───────────────
AUDIO_GYM_LEADER_BATTLE  = 0xEA
AUDIO_TRAINER_BATTLE     = 0xED
AUDIO_WILD_BATTLE        = 0xF0
AUDIO_FINAL_BATTLE       = 0xF3   # champion
AUDIO_DEFEATED_TRAINER   = 0xF6
AUDIO_DEFEATED_WILD      = 0xF9
AUDIO_DEFEATED_GYM       = 0xFC

# ── Map IDs → human-readable names ───────────────────────────────────────────
MAP_NAMES: dict[int, str] = {
    # Towns / Cities
    0:   "Pallet Town",
    1:   "Viridian City",
    2:   "Pewter City",
    3:   "Cerulean City",
    9:   "Indigo Plateau",
    # Routes
    12:  "Route 1",
    13:  "Route 2",
    14:  "Route 3",
    15:  "Route 4",
    33:  "Route 22",
    34:  "Route 23",
    35:  "Route 24",
    36:  "Route 25",
    # Pallet Town buildings
    37:  "Red House 1F",
    38:  "Red House 2F",
    39:  "Blue House",
    40:  "Oak Lab",
    # Viridian City buildings
    41:  "Viridian Pokecenter",
    42:  "Viridian Mart",
    43:  "Viridian School",
    44:  "Name Rater",
    45:  "Viridian Gym",
    # Route 22
    193: "Route 22 Gate",
    # Route 2 / Viridian Forest
    46:  "Route 2 Gate",
    47:  "Viridian Forest N Gate",
    48:  "Route 2 Trade House",
    49:  "Route 2 Diglett Cave",
    50:  "Viridian Forest S Gate",
    51:  "Viridian Forest",
    # Pewter City
    52:  "Pewter Museum 1F",
    53:  "Pewter Museum 2F",
    54:  "Pewter Gym",
    55:  "Pewter Nidoran House",
    56:  "Pewter Mart",
    57:  "Pewter Speech House",
    58:  "Pewter Pokecenter",
    # Route 3 / Mt Moon
    59:  "Mt Moon Pokecenter",
    60:  "Mt Moon 1F",
    61:  "Mt Moon B1F",
    62:  "Mt Moon B2F",
    # Cerulean City
    63:  "Cerulean Pokecenter",    # ◆ verify ID
    64:  "Cerulean Gym",
    65:  "Bike Shop",              # ◆ verify ID
    66:  "Cerulean Mart",          # ◆ verify ID
    67:  "Cerulean Badge House",   # ◆ verify ID
    68:  "Cerulean Trashed House", # ◆ verify ID
    69:  "Cerulean Trade House",   # ◆ verify ID
    # Route 24 / 25
    199: "Bill's House",           # ◆ verify ID
    # Cerulean Cave (Unknown Dungeon)
    196: "Cerulean Cave 1F",       # ◆ verify ID
    197: "Cerulean Cave 2F",       # ◆ verify ID
    198: "Cerulean Cave B1F",      # ◆ verify ID
}

# ── Map sets ──────────────────────────────────────────────────────────────────
POKECENTER_MAPS: frozenset[int] = frozenset({
    41,   # Viridian City
    58,   # Pewter City
    59,   # Mt Moon (Route 3)
    63,   # Cerulean City      ◆ verify ID
    69,   # Vermilion City     ◆ verify
    78,   # Lavender Town      ◆ verify
    82,   # Celadon City       ◆ verify
    91,   # Fuchsia City       ◆ verify
    96,   # Cinnabar Island    ◆ verify
    101,  # Saffron City       ◆ verify
})

TRANSIT_MAPS: frozenset[int] = frozenset({
    47,   # Viridian Forest N Gate
    50,   # Viridian Forest S Gate
    193,  # Route 22 Gate
})

GYM_MAPS: frozenset[int] = frozenset({
    45,   # Viridian Gym
    54,   # Pewter Gym
    64,   # Cerulean Gym
})

# ── Tile reward multiplier per map ────────────────────────────────────────────
MAP_TILE_MULT: dict[int, float] = {
    37:  0.5,   # Red House 1F
    38:  0.5,   # Red House 2F
    39:  0.5,   # Blue House
    40:  0.5,   # Oak Lab
    0:   0.8,   # Pallet Town
    12:  1.0,   # Route 1  (baseline)
    1:   1.2,   # Viridian City
    41:  1.2,   # Viridian Pokecenter
    42:  1.2,   # Viridian Mart
    43:  1.2,   # Viridian School
    44:  1.2,   # Name Rater
    45:  1.5,   # Viridian Gym
    33:  1.5,   # Route 22
    193: 1.5,   # Route 22 Gate
    46:  1.8,   # Route 2 Gate
    48:  1.8,   # Route 2 Trade House
    49:  1.8,   # Route 2 Diglett Cave
    50:  1.8,   # Viridian Forest S Gate
    51:  2.0,   # Viridian Forest
    47:  2.0,   # Viridian Forest N Gate
    2:   2.5,   # Pewter City
    58:  2.5,   # Pewter Pokecenter
    56:  2.5,   # Pewter Mart
    57:  2.5,   # Pewter Speech House
    55:  2.5,   # Pewter Nidoran House
    52:  2.8,   # Pewter Museum 1F
    53:  2.8,   # Pewter Museum 2F
    54:  3.0,   # Pewter Gym
    14:  3.3,   # Route 3
    59:  3.5,   # Mt Moon Pokecenter
    60:  4.0,   # Mt Moon 1F
    61:  4.5,   # Mt Moon B1F
    62:  5.0,   # Mt Moon B2F
    3:   5.5,   # Cerulean City
    63:  5.5,   # Cerulean Pokecenter
    65:  5.5,   # Bike Shop
    66:  5.5,   # Cerulean Mart
    67:  5.5,   # Cerulean Badge House
    68:  5.5,   # Cerulean Trashed House
    69:  5.5,   # Cerulean Trade House
    64:  6.0,   # Cerulean Gym
    35:  6.5,   # Route 24
    36:  7.0,   # Route 25
    199: 7.0,   # Bill's House
    196: 8.0,   # Cerulean Cave 1F
    197: 8.0,   # Cerulean Cave 2F
    198: 8.0,   # Cerulean Cave B1F
    34:  4.0,   # Route 23  (gated, late-game)
    9:   9.0,   # Indigo Plateau
}

# ── Map global origins (tile units, Route 1 top-left = (0,0)) ────────────────
# Derived from pixel coords on kanto_full.png (7200×7200, 16px/tile).
# Anchor: Route 1 TL pixel (1504, 3952).  Formula: gx=round((px-1504)/16), gy=round((py-3952)/16).
MAP_GLOBAL_ORIGIN: dict[int, tuple[int, int]] = {
    # Outdoor / overworld
    0:   (  0,   36),   # Pallet Town
    12:  (  0,    0),   # Route 1  ← ANCHOR
    1:   (-10,  -36),   # Viridian City
    33:  (-50,  -28),   # Route 22
    193: (-62,  -30),   # Route 22 Gate
    13:  (  0, -108),   # Route 2
    34:  (-50, -172),   # Route 23
    9:   (-50, -190),   # Indigo Plateau
    50:  (-12,  -66),   # Viridian Forest S Gate
    46:  (-12,  -76),   # Route 2 Gate
    51:  ( 22,  -88),   # Viridian Forest
    47:  (-12,  -95),   # Viridian Forest N Gate
    2:   (-10, -144),   # Pewter City
    14:  ( 30, -136),   # Route 3
    15:  ( 80, -154),   # Route 4
    3:   (170, -162),   # Cerulean City
    35:  (180, -198),   # Route 24
    36:  (200, -198),   # Route 25
    # Pallet Town buildings
    37:  (-10,   34),   # Red House 1F
    38:  (-20,   34),   # Red House 2F
    39:  ( 22,   34),   # Blue House
    40:  ( 22,   44),   # Oak Lab
    # Viridian City buildings
    41:  ( 32,   -8),   # Viridian Pokecenter
    42:  ( 32,  -18),   # Viridian Mart
    43:  (-20,  -46),   # Viridian School
    44:  (-10,  -46),   # Name Rater
    45:  ( 32,  -38),   # Viridian Gym
    # Route 2 area
    48:  ( 22,  -98),   # Route 2 Trade House
    49:  ( 22, -108),   # Route 2 Diglett Cave
    # Pewter City buildings
    54:  (-22, -140),   # Pewter Gym
    58:  (-16, -106),   # Pewter Pokecenter
    57:  (-20, -122),   # Pewter Speech House
    56:  ( 16, -154),   # Pewter Mart
    55:  ( 26, -154),   # Pewter Nidoran House
    52:  ( -6, -154),   # Pewter Museum 1F
    53:  ( -6, -158),   # Pewter Museum 2F
    # Route 3 / Mt Moon
    59:  ( 64, -156),   # Mt Moon Pokecenter
    60:  ( 84, -192),   # Mt Moon 1F
    61:  ( 54, -192),   # Mt Moon B1F
    62:  ( 12, -192),   # Mt Moon B2F
    # Cerulean City buildings
    63:  (212, -164),   # Cerulean Pokecenter   ◆ verify ID
    64:  (212, -134),   # Cerulean Gym
    65:  (170, -124),   # Bike Shop             ◆ verify ID
    66:  (202, -124),   # Cerulean Mart         ◆ verify ID
    67:  (160, -164),   # Cerulean Badge House  ◆ verify ID
    68:  (202, -172),   # Cerulean Trashed House◆ verify ID
    69:  (170, -173),   # Cerulean Trade House  ◆ verify ID
    # Route 24 / 25
    199: (242, -208),   # Bill's House          ◆ verify ID
    # Cerulean Cave (Unknown Dungeon)
    196: (128, -194),   # Cerulean Cave 1F      ◆ verify ID
    197: (128, -214),   # Cerulean Cave 2F      ◆ verify ID
    198: (128, -174),   # Cerulean Cave B1F     ◆ verify ID
}

# ── Map width in tiles (east-west, from pixel LR data) ───────────────────────
# Formula: round((LR_x - TL_x) / 16).
MAP_SIZE: dict[int, int] = {
    # Towns / Cities
    0:   20,   # Pallet Town           (1824-1504)/16
    1:   40,   # Viridian City         (1984-1344)/16
    2:   40,   # Pewter City           (1984-1344)/16
    3:   40,   # Cerulean City         (4864-4224)/16
    9:   20,   # Indigo Plateau        (1024-704)/16
    # Routes
    12:  20,   # Route 1               (1824-1504)/16
    13:  20,   # Route 2               (1824-1504)/16
    14:  70,   # Route 3               (3104-1984)/16
    15:  90,   # Route 4               (4224-2784)/16
    33:  40,   # Route 22              (1344-704)/16
    34:  20,   # Route 23              (1024-704)/16
    35:  20,   # Route 24              (4704-4384)/16
    36:  60,   # Route 25              (5664-4704)/16
    # Pallet Town buildings
    37:   8,   # Red House 1F          (1472-1344)/16
    38:   8,   # Red House 2F          (1312-1184)/16
    39:   8,   # Blue House            (1984-1856)/16
    40:  10,   # Oak Lab               (2016-1856)/16
    # Viridian City buildings
    41:  14,   # Viridian Pokecenter   (2238-2016)/16 ≈ 14
    42:   8,   # Viridian Mart         (2144-2016)/16
    43:   8,   # Viridian School       (1312-1184)/16
    44:   8,   # Name Rater            (1472-1344)/16
    45:  20,   # Viridian Gym          (2336-2016)/16
    193: 10,   # Route 22 Gate         (672-512)/16
    # Route 2 / Viridian Forest
    46:  10,   # Route 2 Gate          (1472-1312)/16
    47:  10,   # Viridian Forest N Gate(1472-1312)/16
    48:   8,   # Route 2 Trade House   (1984-1856)/16
    49:   8,   # Route 2 Diglett Cave  (1984-1856)/16
    50:  10,   # Viridian Forest S Gate(1472-1312)/16
    51:  34,   # Viridian Forest       (2400-1856)/16
    # Pewter City buildings
    52:  20,   # Pewter Museum 1F      (1728-1408)/16
    53:  14,   # Pewter Museum 2F      (1632-1408)/16
    54:  10,   # Pewter Gym            (1312-1152)/16
    55:   8,   # Pewter Nidoran House  (2048-1920)/16
    56:   8,   # Pewter Mart           (1888-1760)/16
    57:   8,   # Pewter Speech House   (1312-1184)/16
    58:  14,   # Pewter Pokecenter     (1472-1248)/16
    # Route 3 / Mt Moon
    59:  14,   # Mt Moon Pokecenter    (2752-2528)/16
    60:  40,   # Mt Moon 1F            (3488-2848)/16
    61:  28,   # Mt Moon B1F           (2816-2368)/16
    62:  40,   # Mt Moon B2F           (2336-1696)/16
    # Cerulean City buildings
    63:  14,   # Cerulean Pokecenter   (5120-4896)/16    ◆ verify ID
    64:  10,   # Cerulean Gym          (5056-4896)/16
    65:   8,   # Bike Shop             (4352-4224)/16    ◆ verify ID
    66:   8,   # Cerulean Mart         (4864-4736)/16    ◆ verify ID
    67:   8,   # Cerulean Badge House  (4192-4064)/16    ◆ verify ID
    68:   8,   # Cerulean Trashed House(4864-4736)/16    ◆ verify ID
    69:   8,   # Cerulean Trade House  (4352-4224)/16    ◆ verify ID
    # Route 24 / 25
    199:  8,   # Bill's House          (5504-5376)/16    ◆ verify ID
    # Cerulean Cave
    196: 30,   # Cerulean Cave 1F      (4032-3552)/16    ◆ verify ID
    197: 30,   # Cerulean Cave 2F      (4032-3552)/16    ◆ verify ID
    198: 30,   # Cerulean Cave B1F     (4032-3552)/16    ◆ verify ID
}

# ── Map height in tiles (north-south, from pixel LR data) ────────────────────
# Formula: round((LR_y - TL_y) / 16).
MAP_HEIGHT: dict[int, int] = {
    # Towns / Cities
    0:   18,   # Pallet Town           (4816-4528)/16
    1:   36,   # Viridian City         (3952-3376)/16
    2:   36,   # Pewter City           (2224-1648)/16
    3:   36,   # Cerulean City         (1936-1360)/16
    9:   18,   # Indigo Plateau        (1200-912)/16
    # Routes
    12:  36,   # Route 1               (4528-3952)/16
    13:  72,   # Route 2               (3376-2224)/16
    14:  18,   # Route 3               (2064-1776)/16
    15:  18,   # Route 4               (1776-1488)/16
    33:  18,   # Route 22              (3792-3504)/16
    34: 144,   # Route 23              (3504-1200)/16
    35:  36,   # Route 24              (1360-784)/16
    36:  18,   # Route 25              (1072-784)/16
    # Pallet Town buildings
    37:   8,   # Red House 1F          (4624-4496)/16
    38:   8,   # Red House 2F          (4624-4496)/16
    39:   8,   # Blue House            (4624-4496)/16
    40:  12,   # Oak Lab               (4848-4656)/16
    # Viridian City buildings
    41:   8,   # Viridian Pokecenter   (3952-3824)/16
    42:   8,   # Viridian Mart         (3792-3664)/16
    43:   8,   # Viridian School       (3344-3216)/16
    44:   8,   # Name Rater            (3344-3216)/16
    45:  18,   # Viridian Gym          (3632-3344)/16
    193:  8,   # Route 22 Gate         (3600-3472)/16
    # Route 2 / Viridian Forest
    46:   8,   # Route 2 Gate          (2864-2736)/16
    47:   8,   # Viridian Forest N Gate(2560-2432)/16
    48:   8,   # Route 2 Trade House   (2512-2384)/16
    49:   8,   # Route 2 Diglett Cave  (2352-2224)/16
    50:   8,   # Viridian Forest S Gate(3024-2896)/16
    51:  48,   # Viridian Forest       (3312-2544)/16
    # Pewter City buildings
    52:   8,   # Pewter Museum 1F      (1616-1488)/16
    53:   2,   # Pewter Museum 2F      (1456-1416)/16
    54:  14,   # Pewter Gym            (1936-1712)/16
    55:   8,   # Pewter Nidoran House  (1616-1488)/16
    56:   8,   # Pewter Mart           (1616-1488)/16
    57:   8,   # Pewter Speech House   (2128-2000)/16
    58:   8,   # Pewter Pokecenter     (2384-2256)/16
    # Route 3 / Mt Moon
    59:   8,   # Mt Moon Pokecenter    (1584-1456)/16
    60:  36,   # Mt Moon 1F            (1456-880)/16
    61:  28,   # Mt Moon B1F           (1328-880)/16
    62:  36,   # Mt Moon B2F           (1456-880)/16
    # Cerulean City buildings
    63:   8,   # Cerulean Pokecenter   (1456-1328)/16   ◆ verify ID
    64:  14,   # Cerulean Gym          (2032-1816)/16 ≈ 14
    65:   8,   # Bike Shop             (2096-1968)/16   ◆ verify ID
    66:   8,   # Cerulean Mart         (2096-1968)/16   ◆ verify ID
    67:   8,   # Cerulean Badge House  (1456-1328)/16   ◆ verify ID
    68:   8,   # Cerulean Trashed House(1328-1200)/16   ◆ verify ID
    69:   8,   # Cerulean Trade House  (1312-1184)/16   ◆ verify ID
    # Route 24 / 25
    199:  8,   # Bill's House          (752-624)/16     ◆ verify ID
    # Cerulean Cave
    196: 18,   # Cerulean Cave 1F      (1136-848)/16    ◆ verify ID
    197: 18,   # Cerulean Cave 2F      (816-528)/16     ◆ verify ID
    198: 18,   # Cerulean Cave B1F     (1456-1168)/16   ◆ verify ID
}
