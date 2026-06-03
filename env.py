"""
env.py — Pokemon Red Gymnasium Environment

RECONSTRUCTION NOTE
-------------------
This file was reconstructed from:
  • play.py, record.py, train.py, bc_pretrain.py (intact, 0% zeros)
  • patch_env8.py — contains verbatim before/after code blocks
  • README.md — complete reward table
  • Copilot session-db-strings.txt — extracted checkpoint notes with RAM
    addresses, function signatures, reward values, and state vector layout
  • setup_state.py — confirms RAM addresses (D35E/D361/D362/D163/D356)

Sections marked  # ◆ RECONSTRUCTED  were inferred from the above sources.
Sections marked  # ◆ APPROXIMATE    contain values that need verification.
The overall structure, all reward formulas, RAM addresses, and the exact
distance / door-spam block are taken directly from the source evidence.
"""

from __future__ import annotations

import collections
import math
import os
from typing import Any

import numpy as np
import gymnasium
from gymnasium import spaces

# PyBoy ≥ 2.0
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# ── Actions ───────────────────────────────────────────────────────────────────

#: Named actions for the 8 non-noop buttons (used by record.py / bc_pretrain.py)
ACTIONS: list[str] = ["up", "down", "left", "right", "a", "b", "start", "select"]

# Total action space = 9 (index 0 = noop, indices 1-8 = ACTIONS above)
_N_ACTIONS = 1 + len(ACTIONS)  # 9

# WindowEvent maps: action index → (press, release)
_PRESS: dict[int, Any] = {
    1: WindowEvent.PRESS_ARROW_UP,
    2: WindowEvent.PRESS_ARROW_DOWN,
    3: WindowEvent.PRESS_ARROW_LEFT,
    4: WindowEvent.PRESS_ARROW_RIGHT,
    5: WindowEvent.PRESS_BUTTON_A,
    6: WindowEvent.PRESS_BUTTON_B,
    7: WindowEvent.PRESS_BUTTON_START,
    8: WindowEvent.PRESS_BUTTON_SELECT,
}
_RELEASE: dict[int, Any] = {
    1: WindowEvent.RELEASE_ARROW_UP,
    2: WindowEvent.RELEASE_ARROW_DOWN,
    3: WindowEvent.RELEASE_ARROW_LEFT,
    4: WindowEvent.RELEASE_ARROW_RIGHT,
    5: WindowEvent.RELEASE_BUTTON_A,
    6: WindowEvent.RELEASE_BUTTON_B,
    7: WindowEvent.RELEASE_BUTTON_START,
    8: WindowEvent.RELEASE_BUTTON_SELECT,
}

# ── Play / recording constants ─────────────────────────────────────────────────
IDLE_FRAMES: int    = 24   # extra ticks between actions (animations, text) in play mode
# Gen 1 tile walk = 16 frames; 16 ticks/step → player_moved is correct every step
PLAY_FRAME_SKIP: int = 16  # frame skip factor used in play / record mode (vs training)

# ── State vector ───────────────────────────────────────────────────────────────
STATE_VEC_SIZE: int = 91   # 57 base + 24 per-slot combat stats (surv/ttd/ttk/spd × 6) + 8 stat stages + 2 enemy threat

# ── Screen / minimap sizes ─────────────────────────────────────────────────────
_SCREEN_SIZE: int   = 84
_FRAME_STACK: int   = 4
_MINIMAP_SIZE: int  = 21   # 21×21 visited-tile grid centred on player

# ── RAM addresses, map data, and helpers — imported from single source of truth ─
from ram_map import *   # noqa: F401,F403

# Private aliases used internally and exported for play.py backward-compat
_STAGE_MULT    = STAGE_MULT
_NEUTRAL_STAGE = NEUTRAL_STAGE

# ── Game state accessor helpers ───────────────────────────────────────────────
from game_helpers import (  # noqa: E402
    read_state,
    get_current_map, get_player_pos,
    is_in_battle, is_trainer_battle, is_wild_battle,
    is_in_pokecenter,
    get_lead, get_lead_hp_ratio,
    count_alive_pokemon,
    get_enemy, get_enemy_stages, get_player_stages, enemy_stat_lowered,
    gen1_dmg_per_turn, combat_survivability,
    status_mult,
)

# ── HP thresholds ─────────────────────────────────────────────────────────────
LOW_HP_THRESHOLD:      float = 0.20   # below this → Phase 1 healing penalties + obs flag
LOW_HP_HEAL_REWARD:    float = 30.0   # bonus when HP crosses above LOW_HP_THRESHOLD (potion / nurse)

# ── Pokecenter reward constants ───────────────────────────────────────────────
NURSE_JOY_POS: dict[int, tuple[int, int]] = {
    m: (3, 3) for m in POKECENTER_MAPS
}  # player tile when talking to Nurse Joy — same layout in all Gen 1 PCs
   # derived from Viridian: nurse px(2064,3840)=tile(3,1), player talks from tile(3,3)
NURSE_PROXIMITY_MAX: float = 0.40   # obs-vector normaliser for nurse distance (vec[24])

# ── Battle reward constants ────────────────────────────────────────────────────
DAMAGE_REWARD:  float = 0.7    # per HP dealt × battle_mult × ratio_scale
WIN_BONUS:      float = 15.0   # × battle_mult × start_ratio_scale
DEATH_PENALTY:  float = -80.0  # × inverse_ratio_scale  (also used as blackout penalty)
FROZEN_DEAD_GRACE:    int   = 10    # steps of animation before freeze penalty starts
FROZEN_DEAD_STEP_R:   float = -8.0  # per step frozen in battle after lead HP=0 (press A!)
RUN_WILD_REWARD:  float = 20.0  # for running from a wild battle at low/critical HP
RUN_WILD_CAP:     int   = 10    # max rewarded runs per episode; beyond this it becomes a penalty
RUN_WILD_PENALTY: float = -15.0 # per run after cap (bot is farming low-HP flee loops)
RUN_PENALTY:      float = -15.0  # for running when ratio < 0.7 (fleeing when strong)

# ── Explore reward constants ──────────────────────────────────────────────────
NEW_TILE_REWARD:       float = 1.3
NEW_MAP_REWARD:      float = 8.0
WALL_PENALTY:        float = -0.05
REVISIT_PENALTY:     float = -0.12
IDLE_PENALTY:        float = -0.10  # per step doing noop or non-directional outside battle/menu (must exceed WALL_PENALTY so B-spam never beats trying a direction)
DOOR_SPAM_PENALTY:   float = -3.0
DOOR_SPAM_THRESHOLD: int   = 90    # steps since last map change; < this → spam (normal entry+exit needs ~60-70 steps)

# ── North-corridor distance reward ────────────────────────────────────────────
# Pushes the bot northward from the bottom of Pallet Town toward Viridian City.
# Uses the same global tile coordinate system as multiplay (MAP_GLOBAL_ORIGIN).
# Origin = center-bottom of Pallet Town in global tiles: (gx=10, gy=54).
# Reward fires only on Pallet Town + Route 1 to establish the initial northward push.
# STOPS in Viridian City so the bot explores freely (Route 22 west, Forest north).
_DISTANCE_ACTIVE_MAPS: frozenset[int] = frozenset({
    0, 37, 38, 39, 40,          # Pallet Town + its buildings
    12,                          # Route 1
})
_DISTANCE_ORIGIN: tuple[int, int] = (10, 54)   # global (gx, gy) — center-bottom Pallet
DISTANCE_REWARD_SCALE: float = 1.0              # reward per tile of new northward max

# ── Menu spam / idle ──────────────────────────────────────────────────────────
MENU_SPAM_PENALTY:   float = -5.0  # for reopening Start menu too quickly after closing
MENU_SPAM_THRESHOLD: int   = 40    # steps since last menu close; < this → spam
MENU_IDLE_ONSET:     int   = 2     # steps in menu before idle penalty starts
MENU_IDLE_PENALTY:   float = -0.4  # per step in menu beyond the onset

# ── Battle idle ───────────────────────────────────────────────────────────────
BATTLE_IDLE_PENALTY: float = -0.01
BATTLE_IDLE_GRACE:   int   = 60    # steps before idle penalty starts (allows animation + menu nav)

# ── PC navigation reward/penalty ──────────────────────────────────────────────
# Y-first axis navigation: when needing a heal, reward each step that reduces
# Y-distance to the nearest PC entrance; penalty if it increases.
# Once on the same Y row as the entrance, switch to minimising X distance.
# Replaces the old flat low_hp per-step penalty.
PC_NAV_STEP_R:    float = 5.0   # reward per step closer to PC entrance (overworld)
NURSE_NAV_STEP_R: float = 5.0   # reward per step closer to Nurse Joy (inside PC, Y-first)

# ── Low-PP penalty ────────────────────────────────────────────────────────────
# PP bytes in RAM encode PP-Ups in bits 6-7; actual current PP = byte & 0x3F.
# When total usable PP drops below this threshold the bot can only Struggle (self-damage).
LOW_PP_TOTAL:  int   = 5     # total remaining PP across all lead moves below this = emergency
LOW_PP_PENALTY: float = -0.8  # per step penalty (same weight as low_hp, stacks when both)

# ── Phase 1 new-tile Y boundaries ─────────────────────────────────────────────
# new_tile is only rewarded in Phase 1 for tiles within Route 1 and Viridian City.
# Lower bound: Viridian PC door (global Y = -10) — nothing north of PC is rewarded.
# Upper bound: just before Pallet Town (global Y = 36) — Pallet excluded so bot
# isn't incentivised to run south away from the PC.
PHASE1_TILE_MIN_Y: int = PC_ENTRANCE_GLOBAL[41][1]   # -10 = Viridian PC entrance
PHASE1_TILE_MAX_Y: int = 35                           # Route 1 southern edge (Pallet starts at 36)

# ── Catch / ball rewards ──────────────────────────────────────────────────────
BALL_BUY_REWARD:     float = 10.0   # per ball purchased
BALL_THROW_REWARD:   float = 15.0   # for throwing a ball in a wild battle
CATCH_NEW_REWARD:    float = 150.0  # bonus for a new species joining the party this episode
# duplicate catch → 0 catch bonus (only throw reward of +15)

# ── Progression rewards ────────────────────────────────────────────────────────
GYM_ENTRY_REWARD:    float = 20.0   # first visit to a gym map
BADGE_REWARD:        float = 1000.0
LEVEL_UP_REWARD:     float = 120.0
XP_REWARD_SCALE:     float = 0.07   # per XP point gained
POKECENTER_ARRIVE:   float = 1.0    # once per visit when entering with HP < 25 %
POKECENTER_HEAL:     float = 200.0  # one-shot full-heal bonus / nurse healed

# "Got away safely!" tile sequence in Gen 1 VRAM character encoding
# Scanned from window tile map (0x9C00-0x9E00) to detect successful wild flee
_GOT_AWAY_TILES: bytes = bytes([0x86, 0xAE, 0xB3, 0x7F, 0xA0, 0xB6, 0xA0, 0xB8])  # "Got away"
# "We hope to see you again!" — appears when nurse finishes healing
_WE_HOPE_TILES:  bytes = bytes([0x96, 0xA4, 0x7F, 0xA7, 0xAE, 0xAF, 0xA4])        # "We hope"

# ── Phase multipliers ─────────────────────────────────────────────────────────
# explore_mult: starts at 1.0, decays toward 0.2 as badge count grows
# battle_mult:  starts at 1.0, grows toward 2.5 with badges
_EXPLORE_MULT_START: float = 1.0
_EXPLORE_MULT_END:   float = 0.2
_BATTLE_MULT_START:  float = 1.0
_BATTLE_MULT_END:    float = 2.5

# ── Episode limits ─────────────────────────────────────────────────────────────
MAX_STEPS:    int   = 8_192
SCORE_FLOOR:  float = -500.0
# Stuck reset removed: the stale-map penalty grows without bound and eventually
# forces movement. A stuck reset let high-scoring episodes end early, locking
# in a good return and teaching the bot that corner-camping = success.

# ── Stale-map penalty ──────────────────────────────────────────────────────────
STALE_MAP_MAX:     float = -0.12  # max penalty per step (reached when steps_on_map > onset + ramp; onset/ramp computed dynamically from map_area)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class PokemonRedEnv(gymnasium.Env):
    """
    Gymnasium environment wrapping Pokemon Red via PyBoy.

    Observation space (Dict):
        "screen"  : uint8  (84, 84, 4)  — 4-frame grayscale stack
        "state"   : float32 (91,)       — structured game-state vector
            [0]      lead HP ratio
            [1]      lead level / 100
            [2]      lead status multiplier
            [3-7]    bench slots 2-6 HP ratios
            [8-12]   bench slots 2-6 levels / 100
            [13]     party size / 6
            [14]     badges / 8
            [15-16]  player x, y (normalised)
            [17-35]  misc exploration / battle scalars
            [36-56]  HMs, items, battle info, moves, PP, dialogue
            [57-62]  per-slot survival probability (ttd-based, per slot)
            [63-68]  per-slot turns-to-die (ttd, per slot)
            [69-74]  per-slot turns-to-kill (ttk, per slot)
            [75-80]  per-slot speed advantage vs enemy
            [81-84]  player stat stages Atk/Def/Spd/Spc, normalised (0=neutral)
            [85-88]  enemy  stat stages Atk/Def/Spd/Spc, normalised (0=neutral)
            [89]     threat ratio (enemy_dmg / lead_hp)
            [90]     catchability estimate [0,1]
        "minimap" : uint8  (21, 21, 1)  — visited-tile binary grid centred on player

    Action space: Discrete(9)
        0=noop  1=up  2=down  3=left  4=right  5=A  6=B  7=Start  8=Select
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rom_path:        str   = "Pokemon_Red.gb",
        headless:        bool  = True,
        render_in_headless: bool = False,  # keep screen buffer live while still using null window
        emulation_speed: int   = 0,    # 0 = unlimited
        display_frames:  int   = 0,    # extra idle frames after each action (play mode)
        play_frame_skip: int   = 16,   # frames to tick per action step (16 = 1 tile walk in Gen 1)
        state_path:      str   = "initial_state.state",
        max_steps:       int   = MAX_STEPS,  # curriculum: start small, increase on resume
    ):
        super().__init__()
        self._rom_path        = rom_path
        self._headless        = headless
        self._render_in_headless = render_in_headless
        self._emulation_speed = emulation_speed
        self._display_frames  = display_frames
        self._play_frame_skip = play_frame_skip
        self._state_path      = state_path
        self._max_steps       = max_steps

        # Gymnasium spaces
        self.observation_space = spaces.Dict({
            "screen":  spaces.Box(0, 255, (_SCREEN_SIZE, _SCREEN_SIZE, _FRAME_STACK), dtype=np.uint8),
            "state":   spaces.Box(-np.inf, np.inf, (STATE_VEC_SIZE,), dtype=np.float32),
            "minimap": spaces.Box(0, 255, (_MINIMAP_SIZE, _MINIMAP_SIZE, 1), dtype=np.uint8),
        })
        self.action_space = spaces.Discrete(_N_ACTIONS)

        # PyBoy instance (created lazily so subproc workers don't inherit handles)
        self._pyboy: PyBoy | None = None

        # Episode tracking
        self._frames: collections.deque = collections.deque(
            maxlen=_FRAME_STACK
        )
        self._steps: int = 0
        self._steps_on_map: int = 0
        self._steps_since_map_change: int = 999  # large default = no penalty on first entry
        self._map_start_pos: dict       = {}         # map_id → (x, y) first entry
        self._max_northward_dist: float = 0.0       # furthest north in global tiles (for distance reward)
        self._prev_nav_pos: tuple[int,int] | None = None   # unified Y-first nav tracker (forest local or global)
        self._min_pc_dist: tuple = (float("inf"), float("inf"))  # Y-first best approach to target PC

        self._visited_coords: set         = set()
        self._visited_gyms:   set         = set()
        self._visited_maps:   set         = set()
        self._stale_steps:    int         = 0      # steps without new tile (stuck timeout)

        self._prev_state: dict            = {}
        self._ep_reward:  float           = 0.0
        self._ep_breakdown: dict[str, float] = {}

        # Respawn exploit guard (set on death, clears nav state on respawn map)
        self._just_respawned: bool = False

        # Phase 1 curriculum: guide bot to PC first before unlocking exploration rewards
        self._phase1_done: bool = False

        # Battle tracking
        self._in_battle_prev:   bool  = False
        self._battle_start_strength: float = 1.0
        self._is_trainer_battle: bool = False
        self._live_strength_ratio: float = 1.0
        self._prev_enemy_hp: int = 0
        self._prev_enemy_stages: dict = {}
        self._prev_party_xp: list[int] = []
        self._prev_party_level: list[int] = []
        self._battle_idle_steps: int = 0
        self._ran_from_battle: bool = False  # set when VRAM shows "Got away safely!"
        self._run_wild_count:  int  = 0      # runs from wild battles this episode (capped at RUN_WILD_CAP)
        self._frozen_dead_steps: int = 0     # steps in battle with lead HP=0 (freeze exploit counter)
        self._nurse_healed_vram: bool = False  # set when VRAM shows "We hope to see you again!"

        # Pokecenter tracking
        self._prev_lead_hp: int = 0
        self._nurse_a_done: set = set()                  # map_ids whose first-visit A reward is exhausted
        self._prev_nurse_nav_pos: tuple[int,int] | None = None  # local pos last step (for Y-first nurse nav)
        self._min_nurse_dist: tuple = (float("inf"), float("inf"))               # Y-first best approach to Nurse Joy this PC visit
        self._visited_pokecenters: set = set()           # map_ids visited at least once
        self._pc_heal_cooldown: dict[int, int] = {}      # map_id → step of last heal
        self._heal_once_rewarded: bool = False           # once-per-episode heal reward

        # Ball / catch tracking
        self._prev_ball_count: int = 0
        self._prev_party_species: frozenset = frozenset()
        self._threw_ball: bool = False
        self._overworld_menu_open: bool = False  # True while Start menu is displayed
        self._menu_open_steps:     int  = 0      # consecutive steps inside the menu
        self._steps_since_menu_close: int = 999  # for spam detection

        # Item / dialogue novelty tracking
        self._prev_total_items: int = 0          # total bag slots in use (for new-item reward)

        # Record mode
        self._record_action_override: int = -1  # set by record.py

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_pyboy(self) -> None:
        """Lazily create the PyBoy instance."""
        if self._pyboy is not None:
            return
        window = "null" if self._headless else "SDL2"
        self._pyboy = PyBoy(self._rom_path, window=window)
        self._pyboy.set_emulation_speed(self._emulation_speed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._ensure_pyboy()

        # Load the saved initial game state
        if os.path.exists(self._state_path):
            with open(self._state_path, "rb") as f:
                self._pyboy.load_state(f)
        else:
            # No state file — just let the game run from wherever it is
            pass

        # Reset episode variables
        self._steps                  = 0
        self._steps_on_map           = 0
        self._steps_since_map_change = 999
        self._map_start_pos          = {}
        self._max_northward_dist     = 0.0
        self._prev_nav_pos               = None
        self._min_pc_dist                = (float("inf"), float("inf"))

        self._visited_coords  = set()
        self._visited_gyms    = set()
        self._visited_maps    = set()
        self._stale_steps     = 0

        self._ep_reward       = 0.0
        self._ep_breakdown    = {}

        self._in_battle_prev         = False
        self._battle_start_strength  = 1.0
        self._is_trainer_battle      = False
        self._live_strength_ratio    = 1.0
        self._prev_enemy_hp          = 0
        self._prev_enemy_stages      = {}
        self._prev_party_xp          = []
        self._prev_party_level       = []
        self._battle_idle_steps      = 0

        self._prev_lead_hp           = 0
        self._nurse_a_done           = set()
        self._prev_nurse_nav_pos     = None
        self._min_nurse_dist     = (float("inf"), float("inf"))
        self._visited_pokecenters    = set()
        self._pc_heal_cooldown       = {}
        self._heal_once_rewarded     = False  # once-per-episode heal reward
        self._just_respawned         = False
        self._phase1_done            = False  # unlocked after first heal
        self._ran_from_battle        = False
        self._run_wild_count         = 0
        self._frozen_dead_steps      = 0
        self._nurse_healed_vram      = False

        self._prev_ball_count  = 0
        self._prev_party_species = frozenset()
        self._threw_ball       = False
        self._overworld_menu_open    = False
        self._menu_open_steps        = 0
        self._steps_since_menu_close = 999

        self._prev_total_items = 0

        self._record_action_override = -1

        # Warm up frame stack with _FRAME_STACK idle ticks.
        # Always render so screen.image is populated (cheap with window="null").
        self._frames.clear()
        for _ in range(_FRAME_STACK):
            self._pyboy.tick(1, True)
            self._frames.append(self._capture_frame())

        state = self._read_state()
        self._prev_state = state

        # Seed starting distance tracking from current position
        m = state["map_id"]
        self._map_start_pos[m]      = (state["player_x"], state["player_y"])
        self._prev_ball_count       = state["items"].get("balls", 0)
        self._prev_total_items      = state["bag_count"]
        self._prev_party_species    = frozenset(
            p["species"] for p in state.get("party", []) if p.get("species", 0) > 0
        )
        self._prev_party_xp         = [p.get("exp", 0) for p in state["party"]]
        self._prev_party_level      = [p.get("level", 1) for p in state["party"]]

        obs = self._obs(state)
        return obs, {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self._pyboy is not None, "call reset() first"

        # In record mode the human action overrides the model action for reward
        # gating, while the engine step is always noop (SDL2 window handles input)
        effective_action = action
        if self._record_action_override >= 0:
            effective_action = self._record_action_override
            self._record_action_override = -1

        # Advance emulation
        self._do_action(action, effective_action)

        # Capture frame
        self._frames.append(self._capture_frame())
        self._steps += 1

        # Read new state
        state = self._read_state()

        # Compute reward
        reward, breakdown = self._compute_reward(state, self._prev_state, effective_action)
        self._prev_state = state

        # Update accumulators for ep_breakdown
        self._ep_reward += reward
        for k, v in breakdown.items():
            self._ep_breakdown[k] = self._ep_breakdown.get(k, 0.0) + v

        # Termination / truncation
        terminated = False
        truncated  = False
        if self._steps >= self._max_steps:
            truncated = True
        if self._ep_reward < SCORE_FLOOR:
            truncated = True

        obs  = self._obs(state)
        info = {
            "state":            state,
            "reward_breakdown": breakdown,
            "ep_breakdown":     self._ep_breakdown if (terminated or truncated) else {},
        }
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._pyboy is not None:
            try:
                self._pyboy.stop()
            except Exception:
                pass
            self._pyboy = None

    # ─────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────

    @property
    def window_open(self) -> bool:
        """True while the SDL2 window is still open (used by record.py)."""
        if self._pyboy is None or self._headless:
            return False
        try:
            # PyBoy 2.x: tick returns False when window is closed
            # We check by peeking at the screen; if pyboy is stopped, it raises
            return True
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────
    # PyBoy helpers
    # ─────────────────────────────────────────────────────────────────────

    def _do_action(self, action: int, effective_action: int) -> None:
        """Press the button for `action`, tick for play_frame_skip frames, release."""
        render_window = (not self._headless) or self._render_in_headless
        n = self._play_frame_skip
        if action != 0:
            self._pyboy.send_input(_PRESS[action])
        for i in range(n):
            # Always render the last frame so screen.image is fresh for _capture_frame().
            # With window="null" this is cheap (renders to an internal buffer, no display).
            # Skipping render on intermediate frames keeps training fast.
            self._pyboy.tick(1, render_window or (i == n - 1))
        if action != 0:
            self._pyboy.send_input(_RELEASE[action])
        # Extra idle frames for play / record mode (so animations are visible)
        for _ in range(self._display_frames):
            self._pyboy.tick(1, render_window)

    def _capture_frame(self) -> np.ndarray:
        """Return a (84, 84) uint8 grayscale frame."""
        import PIL.Image
        img = self._pyboy.screen.image          # PIL RGBA
        gray = img.convert("L").resize(
            (_SCREEN_SIZE, _SCREEN_SIZE), PIL.Image.NEAREST
        )
        return np.array(gray, dtype=np.uint8)

    # ─────────────────────────────────────────────────────────────────────
    # Game state reading — delegated entirely to game_helpers
    # ─────────────────────────────────────────────────────────────────────

    def _read_state(self) -> dict:
        """Read all game state from RAM. Single point of contact with PyBoy memory."""
        return read_state(self._pyboy.memory)


    # ─────────────────────────────────────────────────────────────────────
    # Observation building
    # ─────────────────────────────────────────────────────────────────────

    def _build_state_vec(self, state: dict) -> np.ndarray:
        """Build the 91-dim float32 state vector (see class docstring for layout)."""
        vec = np.zeros(STATE_VEC_SIZE, dtype=np.float32)

        party    = state["party"]
        badges   = state["badges"]
        map_id   = state["map_id"]
        in_b     = state["in_battle"] > 0
        enemy    = state["enemy"]
        items    = state["items"]
        n_badges = bin(badges).count("1")

        lead     = party[0] if party else None

        # Phase multipliers
        t = n_badges / 8.0
        explore_mult = _EXPLORE_MULT_START + t * (_EXPLORE_MULT_END - _EXPLORE_MULT_START)
        battle_mult  = _BATTLE_MULT_START  + t * (_BATTLE_MULT_END  - _BATTLE_MULT_START)

        # Nurse proximity (used in state vec)
        in_pc = map_id in POKECENTER_MAPS
        nurse_pos = NURSE_JOY_POS.get(map_id)
        if in_pc and nurse_pos:
            nurse_dist = abs(state["player_x"] - nurse_pos[0]) + abs(state["player_y"] - nurse_pos[1])
            nurse_prox = max(0.0, 1.0 - nurse_dist / 12.0) * NURSE_PROXIMITY_MAX
        else:
            nurse_dist = 99
            nurse_prox = 0.0

        # Lead pokemon
        lead_hp_r    = (lead["hp"] / lead["max_hp"])   if lead else 0.0
        lead_level   = lead["level"]                    if lead else 1
        lead_status  = lead["status"]                   if lead else 0

        # ── Indices 0-35 ─────────────────────────────────────────────────
        vec[0]  = lead_hp_r
        vec[1]  = lead_level / 100.0
        vec[2]  = status_mult(lead_status)
        # Party slots 2-6 HP ratios
        for i in range(1, 6):
            p       = party[i] if i < len(party) else None
            vec[3 + i - 1] = (p["hp"] / p["max_hp"]) if p else 0.0
        # Party slots 2-6 levels
        for i in range(1, 6):
            p       = party[i] if i < len(party) else None
            vec[8 + i - 1] = (p["level"] / 100.0) if p else 0.0
        vec[13] = len(party) / 6.0
        vec[14] = n_badges / 8.0
        vec[15] = state["player_x"] / 20.0
        vec[16] = state["player_y"] / 20.0
        vec[17] = self._steps_on_map / 512.0
        vec[18] = self._steps_since_map_change / 512.0
        vec[19] = explore_mult
        vec[20] = battle_mult
        vec[21] = len(self._visited_coords) / 2000.0
        vec[22] = self._steps / max(self._max_steps, 1)
        vec[23] = float(in_pc)
        vec[24] = nurse_prox / NURSE_PROXIMITY_MAX
        vec[25] = float(lead_hp_r < LOW_HP_THRESHOLD)  if lead else 0.0
        vec[26] = float(lead_hp_r < LOW_HP_THRESHOLD and in_b) if lead else 0.0
        vec[27] = items.get("balls", 0) / 10.0
        vec[28] = len(state["pokedex_owned"]) / 151.0
        # XP delta since last step (raw, clipped)
        xp_now  = sum(p.get("exp", 0) for p in party)
        xp_prev = sum(self._prev_party_xp) if self._prev_party_xp else xp_now
        vec[29] = min(max((xp_now - xp_prev) / 1000.0, -1.0), 1.0)
        # Total party HP ratio
        total_hp  = sum(p["hp"] for p in party)
        total_mhp = sum(p["max_hp"] for p in party) or 1
        vec[30] = total_hp / total_mhp
        vec[31] = float(self._is_trainer_battle)
        vec[32] = float(np.clip(self._live_strength_ratio  / 5.0, 0.0, 1.0))
        vec[33] = float(np.clip(self._battle_start_strength / 5.0, 0.0, 1.0))
        vec[34] = len(self._visited_maps) / 17.0
        vec[35] = min(self._stale_steps / max(self._max_steps, 1), 1.0)

        # ── Indices 36-56 (from session notes) ───────────────────────────
        vec[36] = float(items.get("cut",      False))  # HM01 Cut
        vec[37] = float(items.get("strength", False))  # HM04 Strength
        vec[38] = float(items.get("surf",     False))  # HM03 Surf
        vec[39] = items.get("heals", 0) / 10.0
        vec[40] = map_id / 256.0
        vec[41] = float(in_b)
        if enemy:
            vec[42] = enemy["hp"] / enemy["max_hp"]
            vec[43] = enemy["species"] / 151.0
            vec[44] = enemy["level"] / 100.0
        if lead:
            moves = lead.get("moves", [0, 0, 0, 0])
            pps   = lead.get("pp",    [0, 0, 0, 0])
            for j in range(4):
                vec[45 + j] = moves[j] / 166.0
                max_pp = 40
                vec[49 + j] = pps[j] / max_pp
        vec[53] = float(state["dialogue"])
        vec[54] = state.get("npc_talk_flag", 0) / 20.0
        vec[55] = status_mult(enemy["status"]) if enemy else 1.0
        vec[56] = status_mult(lead_status)     if lead  else 1.0

        # ── Indices 57-80: per-slot combat stats vs current enemy ─────────
        # surv[57-62]  adj_ttd[63-68]  adj_ttk[69-74]  spd_adv[75-80]
        # Lead (slot 0) uses live stat stages; bench slots use neutral stages
        # (stages reset on switch-in). Zero when not in battle or no enemy.
        stages   = state.get("stages", {})
        ps       = stages.get("player", {})
        es       = stages.get("enemy",  {})
        if in_b and enemy:
            e_atk_m = _STAGE_MULT[max(0, min(12, es.get("atk", _NEUTRAL_STAGE) - 1))]
            e_def_m = _STAGE_MULT[max(0, min(12, es.get("def", _NEUTRAL_STAGE) - 1))]
            e_spd_m = _STAGE_MULT[max(0, min(12, es.get("spd", _NEUTRAL_STAGE) - 1))]
        for i in range(6):
            p = party[i] if i < len(party) else None
            if in_b and enemy and p and p["hp"] > 0:
                if i == 0:
                    p_atk_m = _STAGE_MULT[max(0, min(12, ps.get("atk", _NEUTRAL_STAGE) - 1))]
                    p_def_m = _STAGE_MULT[max(0, min(12, ps.get("def", _NEUTRAL_STAGE) - 1))]
                    p_spd_m = _STAGE_MULT[max(0, min(12, ps.get("spd", _NEUTRAL_STAGE) - 1))]
                else:
                    p_atk_m = p_def_m = p_spd_m = 1.0   # stages reset on switch-in
                ttd_i, ttk_i, surv_i = combat_survivability(
                    p["hp"],
                    gen1_dmg_per_turn(p["level"], p["atk_stat"], p_atk_m, p["status"],
                                       enemy["def_stat"], e_def_m,
                                       enemy["status"], enemy["max_hp"]),
                    p["spd_stat"] * p_spd_m,
                    enemy["hp"],
                    gen1_dmg_per_turn(enemy["level"], enemy["atk_stat"], e_atk_m, enemy["status"],
                                       p["def_stat"], p_def_m,
                                       p["status"], p["max_hp"]),
                    enemy["spd_stat"] * e_spd_m,
                )
                denom     = max(p["spd_stat"] * p_spd_m + enemy["spd_stat"] * e_spd_m, 1.0)
                spd_adv_i = (p["spd_stat"] * p_spd_m - enemy["spd_stat"] * e_spd_m) / denom
            else:
                surv_i = ttd_i = ttk_i = spd_adv_i = 0.0
            vec[57 + i] = float(np.clip(surv_i  / 5.0,  0.0, 1.0))
            vec[63 + i] = float(np.clip(ttd_i   / 10.0, 0.0, 1.0))
            vec[69 + i] = float(np.clip(ttk_i   / 10.0, 0.0, 1.0))
            vec[75 + i] = float(np.clip(spd_adv_i, -1.0, 1.0))

        # ── Indices 81-84: player in-battle stat stages (Atk/Def/Spd/Spc) ──
        # ── Indices 85-88: enemy  in-battle stat stages (Atk/Def/Spd/Spc) ──
        # Normalised: (stage_value − 7) / 6 → [−1.17, +1.0]; 0 = neutral.
        in_b_flag = float(state["in_battle"] > 0)
        for j, key in enumerate(("atk", "def", "spd", "spc")):
            vec[81 + j] = ((ps.get(key, _NEUTRAL_STAGE) - _NEUTRAL_STAGE) / 6.0) * in_b_flag
            vec[85 + j] = ((es.get(key, _NEUTRAL_STAGE) - _NEUTRAL_STAGE) / 6.0) * in_b_flag

        # ── Indices 89-90: enemy threat & catchability ────────────────────
        # vec[89]: enemy absolute strength — offensive power × remaining HP fraction
        #          = (atk_eff / 255) × hp_ratio  → [0, 1], degrades as enemy is damaged.
        # vec[90]: Gen-1 catch probability for a standard Poké Ball → [0, 1].
        #          Zero outside of wild battle. Rewards bot for weakening + statusing.
        if in_b and enemy and state.get("in_battle") == 1:   # wild battle only
            e_atk_eff = enemy["atk_stat"] * _STAGE_MULT[max(0, min(12, es.get("atk", _NEUTRAL_STAGE) - 1))]
            hp_frac   = enemy["hp"] / max(enemy["max_hp"], 1)
            vec[89]   = float(np.clip((e_atk_eff / 255.0) * hp_frac, 0.0, 1.0))
            vec[90]   = float(catch_probability(
                enemy.get("catch_rate", 45),
                enemy["hp"], enemy["max_hp"], enemy["status"],
            ))

        return vec

    def _build_minimap(self, state: dict) -> np.ndarray:
        """Build a 21×21 uint8 binary grid of visited tiles centred on player."""
        grid = np.zeros((_MINIMAP_SIZE, _MINIMAP_SIZE), dtype=np.uint8)
        cx   = state["player_x"]
        cy   = state["player_y"]
        half = _MINIMAP_SIZE // 2  # 10
        cur_map = state["map_id"]
        for (mid, mx, my) in self._visited_coords:
            if mid != cur_map:
                continue
            dx = mx - cx + half
            dy = my - cy + half
            if 0 <= dx < _MINIMAP_SIZE and 0 <= dy < _MINIMAP_SIZE:
                grid[dy, dx] = 255
        # Mark current player tile
        grid[half, half] = 128
        return grid[:, :, np.newaxis]  # (21, 21, 1)

    def _obs(self, state: dict) -> dict:
        """Build the full observation dict."""
        screen_stack = np.stack(list(self._frames), axis=-1)  # (84, 84, 4)
        return {
            "screen":  screen_stack,
            "state":   self._build_state_vec(state),
            "minimap": self._build_minimap(state),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Reward computation
    # ─────────────────────────────────────────────────────────────────────

    def _phase_mults(self, badges: int) -> tuple[float, float]:
        t = bin(badges).count("1") / 8.0
        em = _EXPLORE_MULT_START + t * (_EXPLORE_MULT_END - _EXPLORE_MULT_START)
        bm = _BATTLE_MULT_START  + t * (_BATTLE_MULT_END  - _BATTLE_MULT_START)
        return em, bm

    def _compute_reward(
        self, state: dict, prev: dict, action: int
    ) -> tuple[float, dict[str, float]]:
        reward: float = 0.0
        bd: dict[str, float] = {}

        badges        = state["badges"]
        explore_mult, battle_mult = self._phase_mults(badges)
        party         = state["party"]
        lead          = get_lead(state)
        in_battle     = is_in_battle(state)
        map_id        = get_current_map(state)
        player_x, player_y = get_player_pos(state)
        enemy         = get_enemy(state)

        prev_party    = prev.get("party", [])
        prev_lead     = get_lead(prev)
        prev_in_battle = is_in_battle(prev)
        # True only when the player physically moved this step
        player_moved = (player_x != prev.get("player_x", player_x) or
                        player_y != prev.get("player_y", player_y))

        # ── Overworld menu state tracking + spam/idle penalties ──────────
        # Auto-close: battle, map change, or player physically moved (can't move in menu).
        if in_battle or map_id != prev.get("map_id", map_id) or player_moved:
            if self._overworld_menu_open:
                self._steps_since_menu_close = 0
            self._overworld_menu_open = False
            self._menu_open_steps     = 0
        elif action == 7 and not in_battle:
            # Opening the Start menu
            if not self._overworld_menu_open:
                # Spam check: reopened too soon after last close
                if self._steps_since_menu_close < MENU_SPAM_THRESHOLD:
                    reward += MENU_SPAM_PENALTY
                    bd["menu_spam"] = bd.get("menu_spam", 0.0) + MENU_SPAM_PENALTY
            self._overworld_menu_open = True
        elif action == 6:
            # B press — close menu regardless of submenu depth
            if self._overworld_menu_open:
                self._overworld_menu_open    = False
                self._menu_open_steps        = 0
                self._steps_since_menu_close = 0

        # Track consecutive steps inside the menu and penalise prolonged idling.
        # Skip penalty on the B-press step (closing action) — reward closing, not punish it.
        if self._overworld_menu_open and action != 6:
            self._menu_open_steps += 1
            if self._menu_open_steps > MENU_IDLE_ONSET:
                reward += MENU_IDLE_PENALTY
                bd["menu_idle"] = bd.get("menu_idle", 0.0) + MENU_IDLE_PENALTY
        else:
            self._steps_since_menu_close += 1

        # ── Tile / exploration ────────────────────────────────────────────
        coord = (map_id, player_x, player_y)
        # Route 2 (map 13) — split by local Y (measured from ImageJ, 16px/tile):
        #   player_y > 12 → south segment (Viridian side) → 1.8
        #   player_y ≤ 12 → north segment (Pewter/Diglett side) → 2.2
        if map_id == 13:
            tile_mult = 1.8 if player_y > 12 else 2.2
        # Route 4 (map 15) — split by local X:
        #   player_x < 24 → west segment (Mt Moon side) → 3.5
        #   player_x ≥ 24 → east segment (Cerulean side) → 4.0
        elif map_id == 15:
            tile_mult = 4.0 if player_x >= 24 else 3.5
        else:
            tile_mult = MAP_TILE_MULT.get(map_id, 1.0)
        if coord not in self._visited_coords:
            self._visited_coords.add(coord)
            self._stale_steps = 0
            # Phase 1: only reward tiles south of / at the Viridian PC door (global Y ≥ -10)
            give_tile = True
            if not self._phase1_done:
                if map_id in MAP_GLOBAL_ORIGIN:
                    gy = MAP_GLOBAL_ORIGIN[map_id][1] + player_y
                    give_tile = PHASE1_TILE_MIN_Y <= gy <= PHASE1_TILE_MAX_Y
                else:
                    give_tile = False  # indoor/dungeon maps excluded in Phase 1
            if give_tile:
                reward += NEW_TILE_REWARD * explore_mult * tile_mult
                bd["new_tile"] = bd.get("new_tile", 0.0) + NEW_TILE_REWARD * explore_mult * tile_mult
        else:
            # Count every non-battle step without a new tile (includes noops/standing still)
            if not in_battle:
                self._stale_steps += 1
            if player_moved:
                # Bot moved but landed on an already-visited tile
                reward += REVISIT_PENALTY
                bd["revisit"] = bd.get("revisit", 0.0) + REVISIT_PENALTY

        # ── New map ───────────────────────────────────────────────────────
        prev_map = prev.get("map_id", map_id)
        if map_id not in self._visited_maps and map_id not in TRANSIT_MAPS:
            self._visited_maps.add(map_id)
            reward += NEW_MAP_REWARD * explore_mult
            bd["new_map"]   = bd.get("new_map", 0.0) + NEW_MAP_REWARD * explore_mult

            # Clear respawn flag only when we've reached a new non-PC map.
            if map_id not in POKECENTER_MAPS:
                self._just_respawned = False

        # ── Northward distance reward ──────────────────────────────────────
        # Only on Pallet Town + Route 1 to establish initial northward push.
        # Stops in Viridian City so exploration rewards guide from there.
        if map_id in _DISTANCE_ACTIVE_MAPS and map_id in MAP_GLOBAL_ORIGIN:
            gx = MAP_GLOBAL_ORIGIN[map_id][0] + player_x
            gy = MAP_GLOBAL_ORIGIN[map_id][1] + player_y
            northward = _DISTANCE_ORIGIN[1] - gy   # positive = north of origin
            if northward > self._max_northward_dist:
                delta = northward - self._max_northward_dist
                self._max_northward_dist = northward
                dist_r = delta * DISTANCE_REWARD_SCALE * explore_mult
                reward += dist_r
                bd["distance"] = bd.get("distance", 0.0) + dist_r

        # ── Gym entry ─────────────────────────────────────────────────────
        if map_id in GYM_MAPS and map_id not in self._visited_gyms:
            self._visited_gyms.add(map_id)
            reward += GYM_ENTRY_REWARD
            bd["gym_entry"] = bd.get("gym_entry", 0.0) + GYM_ENTRY_REWARD

        # ── Badge gain ────────────────────────────────────────────────────
        prev_badges = prev.get("badges", 0)
        new_badges  = bin(badges).count("1") - bin(prev_badges).count("1")
        if new_badges > 0:
            reward += BADGE_REWARD * new_badges
            bd["badge"]     = bd.get("badge", 0.0) + BADGE_REWARD * new_badges

        # ── Map distance (patch_env8.py exact code) ───────────────────────
        cur_map = map_id

        if cur_map != prev_map:
            # Door-spam penalty: switching maps twice within 30 steps
            if self._steps_since_map_change < DOOR_SPAM_THRESHOLD:
                reward += DOOR_SPAM_PENALTY
                bd["door_spam"] = bd.get("door_spam", 0.0) + DOOR_SPAM_PENALTY
            self._steps_since_map_change = 0
            self._steps_on_map           = 0
            # Record entry point the FIRST time we visit this map
            if cur_map not in self._map_start_pos:
                self._map_start_pos[cur_map] = (player_x, player_y)
        else:
            self._steps_since_map_change += 1
            if not in_battle:
                self._steps_on_map += 1
                # Stale-map penalty: onset scales with map area (width×height) so
                # large routes allow more exploration before penalising.
                # No cap — penalty grows linearly forever so the bot can never
                # outlast the episode by camping on one map.
                map_area    = MAP_SIZE.get(map_id, 10) * MAP_HEIGHT.get(map_id, 18)
                stale_onset = max(80, map_area // 4)
                stale_ramp  = stale_onset * 2
                if self._steps_on_map > stale_onset:
                    growth  = (self._steps_on_map - stale_onset) / stale_ramp
                    stale_r = STALE_MAP_MAX * growth   # grows without bound
                    reward += stale_r
                    bd["stale_map"] = bd.get("stale_map", 0.0) + stale_r

        # ── Wall penalty (tried to move but didn't) ───────────────────────
        if action in (1, 2, 3, 4):  # directional action
            if not player_moved and not in_battle and not self._overworld_menu_open:
                reward += WALL_PENALTY
                bd["wall"] = bd.get("wall", 0.0) + WALL_PENALTY

        # ── Idle penalty (noop or non-directional while not in battle/menu) ──
        # Discourages standing still or spamming A/B/Select in the overworld.
        if action in (0, 5, 6, 8) and not in_battle and not self._overworld_menu_open:
            reward += IDLE_PENALTY
            bd["idle"] = bd.get("idle", 0.0) + IDLE_PENALTY

        # ── XP / level up ─────────────────────────────────────────────────
        xp_now   = [p.get("exp", 0) for p in party]
        lvl_now  = [p.get("level", 1) for p in party]
        for i, (xp_n, lvl_n) in enumerate(zip(xp_now, lvl_now)):
            if i < len(self._prev_party_xp):
                xp_delta = xp_n - self._prev_party_xp[i]
                if 0 < xp_delta < 10000:
                    xp_r = xp_delta * XP_REWARD_SCALE
                    reward += xp_r
                    bd["xp"] = bd.get("xp", 0.0) + xp_r
            if i < len(self._prev_party_level):
                if lvl_n > self._prev_party_level[i]:
                    reward += LEVEL_UP_REWARD
                    bd["level_up"] = bd.get("level_up", 0.0) + LEVEL_UP_REWARD
        self._prev_party_xp    = xp_now
        self._prev_party_level = lvl_now

        # ── Ball purchase ─────────────────────────────────────────────────
        ball_now = state["items"].get("balls", 0)
        ball_bought = ball_now - self._prev_ball_count
        if ball_bought > 0 and not in_battle:
            reward += BALL_BUY_REWARD * ball_bought
            bd["ball_buy"] = bd.get("ball_buy", 0.0) + BALL_BUY_REWARD * ball_bought
        # Track if we threw a ball (ball count decreased in wild battle)
        if in_battle and is_wild_battle(state):
            if ball_now < self._prev_ball_count:
                self._threw_ball = True
                reward += BALL_THROW_REWARD
                bd["ball_throw"] = bd.get("ball_throw", 0.0) + BALL_THROW_REWARD
        self._prev_ball_count = ball_now

        # ── New item pickup (non-ball) ─────────────────────────────────────
        total_items_now = state["bag_count"]
        if total_items_now > self._prev_total_items and not in_battle:
            new_slots = total_items_now - self._prev_total_items
            reward += 1.0 * new_slots
            bd["new_item"] = bd.get("new_item", 0.0) + 1.0 * new_slots
        self._prev_total_items = total_items_now

        # ── Catch ─────────────────────────────────────────────────────────
        party_species_now = frozenset(
            p["species"] for p in state.get("party", []) if p.get("species", 0) > 0
        )
        new_species = party_species_now - self._prev_party_species
        if new_species:
            reward += CATCH_NEW_REWARD * len(new_species)
            bd["catch"] = bd.get("catch", 0.0) + CATCH_NEW_REWARD * len(new_species)
        self._prev_party_species = party_species_now
        self._threw_ball   = False

        # ── Battle rewards ────────────────────────────────────────────────
        just_entered_battle = in_battle and not prev_in_battle
        just_left_battle    = not in_battle and prev_in_battle

        if just_entered_battle and enemy:
            # Snapshot survivability at battle start (both sides at neutral stages)
            lead_atk = lead["atk_stat"] if lead else 10
            lead_def = lead["def_stat"] if lead else 10
            lead_spd = lead["spd_stat"] if lead else 10
            lead_hp  = lead["hp"]       if lead else 5
            lead_mhp = lead["max_hp"]   if lead else 10
            lead_lvl = lead["level"]    if lead else 5
            lead_st  = lead["status"]   if lead else 0
            _, _, self._battle_start_strength = combat_survivability(
                lead_hp,
                gen1_dmg_per_turn(lead_lvl, lead_atk, 1.0, lead_st,
                                   enemy["def_stat"], 1.0, enemy["status"], enemy["max_hp"]),
                lead_spd * 1.0,
                enemy["hp"],
                gen1_dmg_per_turn(enemy["level"], enemy["atk_stat"], 1.0, enemy["status"],
                                   lead_def, 1.0, lead_st, lead_mhp),
                enemy["spd_stat"] * 1.0,
            )
            self._is_trainer_battle    = is_trainer_battle(state)
            self._live_strength_ratio  = self._battle_start_strength
            self._prev_enemy_hp        = enemy["hp"]
            self._prev_enemy_stages    = dict(state.get("stages", {}).get("enemy", {}))
            self._battle_idle_steps    = 0
            self._frozen_dead_steps    = 0
            self._ran_from_battle      = False   # reset at battle start

        if in_battle and enemy and lead:
            # Live ratio update — include current stat stages
            ps = get_player_stages(state)
            es = get_enemy_stages(state)
            p_atk_m = _STAGE_MULT[max(0, min(12, ps.get("atk", _NEUTRAL_STAGE) - 1))]
            p_def_m = _STAGE_MULT[max(0, min(12, ps.get("def", _NEUTRAL_STAGE) - 1))]
            p_spd_m = _STAGE_MULT[max(0, min(12, ps.get("spd", _NEUTRAL_STAGE) - 1))]
            e_atk_m = _STAGE_MULT[max(0, min(12, es.get("atk", _NEUTRAL_STAGE) - 1))]
            e_def_m = _STAGE_MULT[max(0, min(12, es.get("def", _NEUTRAL_STAGE) - 1))]
            e_spd_m = _STAGE_MULT[max(0, min(12, es.get("spd", _NEUTRAL_STAGE) - 1))]
            _, _, self._live_strength_ratio = combat_survivability(
                lead["hp"],
                gen1_dmg_per_turn(lead["level"], lead["atk_stat"], p_atk_m, lead["status"],
                                   enemy["def_stat"], e_def_m, enemy["status"], enemy["max_hp"]),
                lead["spd_stat"] * p_spd_m,
                enemy["hp"],
                gen1_dmg_per_turn(enemy["level"], enemy["atk_stat"], e_atk_m, enemy["status"],
                                   lead["def_stat"], p_def_m, lead["status"], lead["max_hp"]),
                enemy["spd_stat"] * e_spd_m,
            )

            # Damage reward (HP dealt to enemy this step)
            hp_dealt = self._prev_enemy_hp - enemy["hp"]
            if hp_dealt >= 2:
                # Harder battle (surv_ratio < 1) → bigger reward for landing damage
                ratio_scale = float(np.clip((1.0 / max(self._live_strength_ratio, 0.1)) ** 0.8, 0.1, 1.5))
                dmg_r = DAMAGE_REWARD * hp_dealt * ratio_scale * battle_mult
                reward += dmg_r
                bd["damage"] = bd.get("damage", 0.0) + dmg_r
            self._prev_enemy_hp = enemy["hp"]

            # Battle idle penalty — resets when ≥2 HP dealt OR an enemy stat was lowered.
            # Filters RAM noise from switch animations (hp_dealt threshold ≥2).
            # ALL actions count including A so the bot cannot spam switch/menu to dodge.
            # Tripled when only 1 alive pokemon (switching is impossible — pure spam).
            cur_enemy_stages = get_enemy_stages(state)
            stat_lowered = enemy_stat_lowered(cur_enemy_stages, self._prev_enemy_stages)
            self._prev_enemy_stages = dict(cur_enemy_stages)

            n_alive = count_alive_pokemon(state)
            idle_penalty = BATTLE_IDLE_PENALTY * (4.0 if n_alive <= 1 else 1.0)
            if hp_dealt >= 2 or stat_lowered:
                self._battle_idle_steps = 0
            else:
                self._battle_idle_steps += 1
                if self._battle_idle_steps > BATTLE_IDLE_GRACE:
                    reward += idle_penalty
                    bd["battle_idle"] = bd.get("battle_idle", 0.0) + idle_penalty

            # Lead fainted: fire death penalty immediately on the HP→0 transition.
            # party[0].hp stays 0 so this only triggers once per faint, not every step.
            # Unavoidable — no dialogue freeze can dodge it.
            prev_lead_hp_now = prev_lead["hp"] if prev_lead else 1
            if prev_lead_hp_now > 0 and lead and lead["hp"] == 0:
                reward += DEATH_PENALTY
                bd["death"] = bd.get("death", 0.0) + DEATH_PENALTY
                print("Lead fainted penalty!")

            # Freeze exploit: bot stays in battle with lead HP=0 instead of pressing A.
            # After a short animation grace, apply heavy per-step penalty.
            # -8/step means 10 extra steps = -80 more (another full death penalty).
            if lead and lead["hp"] == 0:
                self._frozen_dead_steps += 1
                if self._frozen_dead_steps > FROZEN_DEAD_GRACE:
                    reward += FROZEN_DEAD_STEP_R
                    bd["frozen_dead"] = bd.get("frozen_dead", 0.0) + FROZEN_DEAD_STEP_R
            else:
                self._frozen_dead_steps = 0

            # VRAM scan: "Got away safely!" — reliable run detection
            vram = bytes(self._pyboy.memory[0x9C00:0x9E00])
            if _GOT_AWAY_TILES in vram:
                self._ran_from_battle = True
        # ── Battle exit rewards ────────────────────────────────────────────
        if just_left_battle:
            prev_lead_hp = prev_lead["hp"]   if prev_lead else 0
            prev_max_hp  = prev_lead["max_hp"] if prev_lead else 1
            self._battle_start_strength = 1.0

            prev_enemy = prev.get("enemy")
            if prev_enemy and prev_enemy["hp"] == 0:
                # Won: enemy fainted
                ratio_scale = float(np.clip(
                    (1.0 / max(self._battle_start_strength, 0.1)) ** 1.2, 0.1, 2.5
                ))
                win_r = WIN_BONUS * ratio_scale * battle_mult
                reward += win_r
                bd["win"] = bd.get("win", 0.0) + win_r
            elif self._ran_from_battle:
                # Confirmed ran away via VRAM "Got away safely!" text
                if not self._is_trainer_battle:
                    prev_hp_ratio = prev_lead_hp / max(prev_max_hp, 1)
                    if prev_hp_ratio < LOW_HP_THRESHOLD:
                        self._run_wild_count += 1
                        if self._run_wild_count <= RUN_WILD_CAP:
                            reward += RUN_WILD_REWARD
                            bd["run_wild"] = bd.get("run_wild", 0.0) + RUN_WILD_REWARD
                        else:
                            reward += RUN_WILD_PENALTY
                            bd["run_wild_pen"] = bd.get("run_wild_pen", 0.0) + RUN_WILD_PENALTY
                    elif self._live_strength_ratio < 0.83 and prev_hp_ratio < LOW_HP_THRESHOLD:
                        reward += RUN_WILD_REWARD
                        bd["run_wild"] = bd.get("run_wild", 0.0) + RUN_WILD_REWARD
                    elif self._live_strength_ratio >= 1.0:
                        reward += RUN_PENALTY
                        bd["run_penalty"] = bd.get("run_penalty", 0.0) + RUN_PENALTY
            else:
                # Not a win, not a confirmed run → blacked out (died)
                ratio_scale = float(np.clip(
                    self._battle_start_strength ** 0.7, 0.5, 2.5
                ))
                death_r = DEATH_PENALTY * ratio_scale
                reward += death_r
                bd["death"] = bd.get("death", 0.0) + death_r
                self._just_respawned = True
                self._nurse_healed_vram = False   # allow VRAM scan to re-trigger at PC
                self._heal_once_rewarded = False  # +500 waiting at PC after every death
                self._prev_nav_pos = None        # reset nav delta on death
                self._prev_nurse_nav_pos = None  # reset nurse nav delta on death
                self._min_nurse_dist     = (float("inf"), float("inf"))
                # _nurse_a_done intentionally NOT reset on death — first-visit reward is once per episode

        # ── PC navigation (Y-first) + Low PP penalty ─────────────────────
        if lead:
            lead_hp_r = get_lead_hp_ratio(state)
            in_pc_now = is_in_pokecenter(state)

            # PP: bits 6-7 of each PP byte = PP-Ups; bits 0-5 = current PP
            lead_pp_total = sum(pp & 0x3F for pp in lead.get("pp", []))
            lead_pp_low   = lead_pp_total < LOW_PP_TOTAL
            # Emergency condition: low HP OR depleted PP
            needs_heal = lead_hp_r < LOW_HP_THRESHOLD or lead_pp_low

            # Low-PP per-step penalty
            if lead_pp_low and not in_pc_now:
                pen = LOW_PP_PENALTY
                reward += pen
                bd["low_pp"] = bd.get("low_pp", 0.0) + pen

            # Y-first PC navigation: Phase 1 only — after first heal the bot
            # navigates back to the PC via self-motivated learning.
            # Only rewards a new personal-best approach distance to prevent
            # the oscillation exploit (go north → reward, south, north → reward again).
            if not self._phase1_done and not in_pc_now:
                cx = cy = tx = ty = None
                if map_id in MAP_GLOBAL_ORIGIN:
                    # Overworld maps: global tile coords → nearest PC entrance
                    cx = MAP_GLOBAL_ORIGIN[map_id][0] + player_x
                    cy = MAP_GLOBAL_ORIGIN[map_id][1] + player_y
                    valid_pcs = [(ex, ey) for _, (ex, ey) in PC_ENTRANCE_GLOBAL.items()
                                 if ex != 0 or ey != 0]
                    if valid_pcs:
                        tx, ty = min(valid_pcs, key=lambda p: abs(cx - p[0]) + abs(cy - p[1]))

                # Reset trackers on map transitions (nav pos only — _min_pc_dist persists
                # across transitions so the bot cannot farm by crossing the Route 1/Pallet boundary)
                if map_id != prev_map:
                    self._prev_nav_pos = None

                if tx is not None:
                    # Y-first: only reward vertical progress until on target row, then horizontal.
                    # X component is pinned to 0 in Y phase so east/west steps give nothing.
                    key = (abs(cy - ty), 0) if cy != ty else (0, abs(cx - tx))
                    if key < self._min_pc_dist:
                        self._min_pc_dist = key
                        reward += PC_NAV_STEP_R
                        bd["approaching_pc"] = bd.get("approaching_pc", 0.0) + PC_NAV_STEP_R
                    self._prev_nav_pos = (cx, cy)
                else:
                    self._prev_nav_pos = None
            elif self._phase1_done or in_pc_now:
                # Clear tracker when Phase 1 is done or inside PC
                self._prev_nav_pos = None

        # ── Pokecenter rewards ────────────────────────────────────────────
        in_pc     = in_pc_now
        nurse_pos = NURSE_JOY_POS.get(map_id)

        if in_pc and nurse_pos and lead:
            lead_hp_r = get_lead_hp_ratio(state)

            # First-visit bonus (suppressed on death-respawn to avoid rewarding dying)
            if map_id not in self._visited_pokecenters:
                self._visited_pokecenters.add(map_id)
                if not self._just_respawned:
                    reward += 3.0
                    bd["pokecenter_visit"] = bd.get("pokecenter_visit", 0.0) + 3.0
                    print("Pokecenter first visit! +3.0")

            # Arrive-at-low-HP bonus
            if prev.get("map_id") not in POKECENTER_MAPS and lead_hp_r < 0.25:
                reward += POKECENTER_ARRIVE
                bd["pokecenter_arrive"] = bd.get("pokecenter_arrive", 0.0) + POKECENTER_ARRIVE

            # Y-first navigation toward Nurse Joy (Phase 1 only)
            if not self._phase1_done:
                nx, ny = nurse_pos
                key = (abs(player_y - ny), 0) if player_y != ny else (0, abs(player_x - nx))
                if key < self._min_nurse_dist:
                    self._min_nurse_dist = key
                    reward += NURSE_NAV_STEP_R
                    bd["nurse_nav"] = bd.get("nurse_nav", 0.0) + NURSE_NAV_STEP_R
                self._prev_nurse_nav_pos = (player_x, player_y)

            # A-press reward: once per PC per episode (first A press only)
            if action == 5 and map_id not in self._nurse_a_done:
                reward += 0.1
                bd["nurse_a"] = bd.get("nurse_a", 0.0) + 0.1
                self._nurse_a_done.add(map_id)  # consume — no more A rewards this map this episode

            # Heal rewards — two independent checks, both with 150-step cooldown.
            prev_party = self._prev_state.get("party", [])
            cur_party  = state.get("party", [])
            prev_lead_pc = prev_party[0] if prev_party else {}
            prev_hr    = prev_lead_pc.get("hp", 0) / max(prev_lead_pc.get("max_hp", 1), 1)
            last_heal  = self._pc_heal_cooldown.get(map_id, -999)
            healed_now = (self._steps - last_heal) > 150

            # 1) Normal heal: lead HP went from below full → full, not a respawn
            if (prev_hr < 1.0 and lead_hp_r == 1.0
                    and healed_now and not self._just_respawned):
                reward += POKECENTER_HEAL
                bd["pokecenter_heal"] = bd.get("pokecenter_heal", 0.0) + POKECENTER_HEAL
                self._pc_heal_cooldown[map_id] = self._steps
                self._prev_nurse_nav_pos = None  # restart nav tracking after heal
                self._min_nurse_dist     = (float("inf"), float("inf"))
                self._phase1_done = True         # unlock exploration rewards
                print(f"Normal heal reward! +{POKECENTER_HEAL:.0f}")

            # 2) Once-per-episode heal: VRAM "We hope to see you again!" = nurse finished healing
            if not self._nurse_healed_vram:
                pc_vram = bytes(self._pyboy.memory[0x9C00:0x9E00])
                if _WE_HOPE_TILES in pc_vram:
                    self._nurse_healed_vram = True
                    if (healed_now or self._just_respawned) and not self._heal_once_rewarded:
                        self._heal_once_rewarded = True
                        heal_once_r = POKECENTER_HEAL + 1300
                        reward += heal_once_r
                        bd["pokecenter_heal_once"] = bd.get("pokecenter_heal_once", 0.0) + heal_once_r
                        self._pc_heal_cooldown[map_id] = self._steps
                        self._prev_nurse_nav_pos = None  # restart nav tracking after heal
                        self._min_nurse_dist     = (float("inf"), float("inf"))
                        self._phase1_done = True         # unlock exploration rewards
                        print(f"Once-per-episode heal reward! +{heal_once_r:.0f}")
        else:
            if prev.get("map_id") in POKECENTER_MAPS:
                prev_pc = prev.get("map_id", -1)
                self._nurse_a_done.add(prev_pc)   # first visit over; no more A rewards on re-entry
                self._prev_nurse_nav_pos = None    # clear nurse nav when leaving PC
                self._min_nurse_dist     = (float("inf"), float("inf"))

        # ── Low-HP recovery reward ────────────────────────────────────────────
        # Reward any transition from below LOW_HP_THRESHOLD to above it (nurse heal,
        # potion, revive) as long as it's not a death-respawn (full HP teleport).
        if not in_battle and not self._just_respawned:
            prev_party_r = self._prev_state.get("party", [])
            prev_lead_r  = prev_party_r[0] if prev_party_r else {}
            prev_hr_r    = prev_lead_r.get("hp", 0) / max(prev_lead_r.get("max_hp", 1), 1)
            if prev_hr_r < LOW_HP_THRESHOLD and lead_hp_r >= LOW_HP_THRESHOLD:
                reward += LOW_HP_HEAL_REWARD
                bd["low_hp_heal"] = bd.get("low_hp_heal", 0.0) + LOW_HP_HEAL_REWARD
                print(f"Low-HP recovery reward! +{LOW_HP_HEAL_REWARD:.0f}")

        # ── Phase 1 curriculum: suppress exploration until first heal ────────
        # Exploration rewards are withheld until the bot has healed at least once
        # per episode. Nav rewards (approaching_pc, nurse_nav) fire unconditionally
        # during Phase 1 so the bot always has a signal toward the PC.
        _PHASE1_REMOVE: frozenset[str] = frozenset({
            "distance", "revisit",
            "new_map",
        })
        if not self._phase1_done:
            reward = sum(v for k, v in bd.items() if k not in _PHASE1_REMOVE)
            bd = {k: v for k, v in bd.items() if k not in _PHASE1_REMOVE}

        return reward, bd

