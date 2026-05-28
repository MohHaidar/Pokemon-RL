"""
multiplay.py — Watch N trained Pokemon Red agents play simultaneously on a shared world map.

RECONSTRUCTION NOTE
-------------------
Reconstructed from:
  • session-db-strings.txt (checkpoint notes): BotThread design, hardlink fix,
    MAP_GLOBAL_ORIGIN import, display-space origin derivation, A/B action display,
    _QUIET set, removed 'dist' column, 17 maps, building rectangles
  • env.py (reconstructed): MAP_GLOBAL_ORIGIN, MAP_SIZE, MAP_NAMES, POKECENTER_MAPS
  • play.py (intact): display format, reward_breakdown keys, info dict structure

Layout
------
  Left  — shared world minimap with all bot positions
  Right — grid of individual bot Game Boy screens with stat overlay

Usage
-----
    python multiplay.py runs/pokemon_ppo_final.zip
    python multiplay.py runs/pokemon_ppo_final.zip --n 8 --speed 2
    python multiplay.py runs/pokemon_ppo_final.zip --n 4 --rom Pokemon_Red.gb

Close the pygame window or press Ctrl+C to stop all bots.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pygame
from sb3_contrib import RecurrentPPO

from env import PokemonRedEnv, ACTIONS
from ram_map import MAP_GLOBAL_ORIGIN, MAP_SIZE, MAP_HEIGHT, MAP_NAMES, POKECENTER_MAPS

# ── Display constants ─────────────────────────────────────────────────────────

_WIN_W: int      = 1280   # fixed window width  (never changes with bot count)
_WIN_H: int      =  800   # fixed window height
_MAP_FRAC: float = 0.62   # world map panel fraction of total width
_SCALE: float    =   4.0  # initial tile scale hint (auto-adjusted to fit panel)
_PADDING: int    =  12    # inner padding for the map panel
_STAT_H: int     =  48    # pixel height of stat text bar below each bot screen (increased for 3 lines)
_SCREEN_NATIVE_W: int = 160
_SCREEN_NATIVE_H: int = 144

# Width and height per map in tiles — sourced from ram_map (derived from LR pixel data).
# _MAP_WIDTH/_MAP_HEIGHT are kept as aliases so the rest of the file is unchanged.
_MAP_WIDTH  = MAP_SIZE
_MAP_HEIGHT = MAP_HEIGHT

# Visited-tile (bright) fill colors and dark silhouette variants
_MAP_FILL: dict[int, tuple[int,int,int]] = {
    0:   ( 60, 120,  60),   # Pallet Town
    1:   ( 60, 120,  60),   # Viridian City
    2:   ( 60, 120,  60),   # Pewter City
    3:   ( 60, 120,  60),   # Cerulean City
    9:   ( 80,  80, 130),   # Indigo Plateau
    12:  ( 50, 110,  50),   # Route 1
    13:  ( 50, 110,  50),   # Route 2
    14:  ( 50, 110,  50),   # Route 3
    15:  ( 50, 110,  50),   # Route 4
    33:  ( 50, 110,  50),   # Route 22
    34:  ( 40, 100,  40),   # Route 23
    35:  ( 50, 110,  50),   # Route 24
    36:  ( 50, 110,  50),   # Route 25
    45:  (160,  50,  50),   # Viridian Gym
    51:  ( 30,  90,  30),   # Viridian Forest
    54:  (160,  50,  50),   # Pewter Gym
    64:  (160,  50,  50),   # Cerulean Gym
    60:  ( 90,  60,  30),   # Mt Moon 1F
    61:  ( 80,  50,  25),   # Mt Moon B1F
    62:  ( 70,  45,  20),   # Mt Moon B2F
    196: ( 85,  55,  30),   # Cerulean Cave 1F
    197: ( 75,  50,  25),   # Cerulean Cave 2F
    198: ( 80,  55,  30),   # Cerulean Cave B1F
}
_DEFAULT_MAP_FILL    = ( 60,  60,  90)
_DEFAULT_MAP_FILL_PC = ( 80,  50, 110)

def _dim(c: tuple[int,int,int]) -> tuple[int,int,int]:
    return (c[0]//4, c[1]//4, c[2]//4)

_MAP_FILL_DARK    = {mid: _dim(col) for mid, col in _MAP_FILL.items()}
_DEFAULT_MAP_FILL_DARK    = _dim(_DEFAULT_MAP_FILL)
_DEFAULT_MAP_FILL_PC_DARK = _dim(_DEFAULT_MAP_FILL_PC)

# Bot colors
_BOT_COLORS: list[tuple[int,int,int]] = [
    (255,  80,  80),
    ( 80, 160, 255),
    ( 80, 255,  80),
    (255, 255,  80),
    (255, 160,  80),
    (200,  80, 255),
    ( 80, 255, 255),
    (255,  80, 200),
]

# Action labels
_ACTION_LABEL: list[str] = ["·"] + ACTIONS

# Reward keys to suppress from the event display
_QUIET: frozenset[str] = frozenset({"battle_idle"})


# ── BotThread ─────────────────────────────────────────────────────────────────

class BotThread(threading.Thread):
    """Runs one headless PokemonRedEnv + RecurrentPPO agent in a background thread.

    State visible to the main thread is updated atomically under ``_lock``.
    Each bot gets a unique hardlink copy of the ROM so PyBoy can create a
    per-bot ``.ram`` save file without conflicts.
    """

    def __init__(
        self,
        bot_id: int,
        model_path: str,
        rom_path: str,
        speed: int = 0,
        state_path: str | None = None,
        max_steps: int = 8_192,
    ) -> None:
        super().__init__(daemon=True)
        self.bot_id     = bot_id
        self.model_path = model_path
        self.rom_path   = rom_path
        self.speed      = speed
        self.state_path = state_path
        self.max_steps  = max_steps

        # ── Shared state (read by main thread) ────────────────────────────
        self.map_id:           int                   = 12
        self.x:                int                   = 0
        self.y:                int                   = 0
        self.last_action:      int                   = 0
        self.ep_reward:        float                 = 0.0
        self.ep_steps:         int                   = 0
        self.reward_breakdown: dict                  = {}
        self.badges:           int                   = 0
        self.lead_level:       int                   = 0
        self.party_size:       int                   = 0
        self.episode:          int                   = 1
        self.in_battle:        bool                  = False
        self.alive:            bool                  = True
        self.error:            str                   = ""
        self.screen_pixels:    np.ndarray | None     = None  # (144, 160, 3) RGB
        self._lock = threading.Lock()

        # ── Per-bot ROM hardlink ───────────────────────────────────────────
        rp           = Path(rom_path)
        self.bot_rom = str(rp.parent / f"{rp.stem}_bot{bot_id}{rp.suffix}")
        self.bot_ram = self.bot_rom + ".ram"

    def run(self) -> None:
        try:
            if not os.path.exists(self.bot_rom):
                os.link(self.rom_path, self.bot_rom)
        except (OSError, NotImplementedError, AttributeError):
            shutil.copy2(self.rom_path, self.bot_rom)
        try:
            self._run_loop()
        except Exception as exc:
            with self._lock:
                self.error = str(exc)
                self.alive = False
        finally:
            for path in [self.bot_rom, self.bot_ram]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

    def _run_loop(self) -> None:
        env   = PokemonRedEnv(
            rom_path=self.bot_rom,
            headless=True,
            render_in_headless=True,
            emulation_speed=self.speed,
            max_steps=self.max_steps,
        )
        model = RecurrentPPO.load(self.model_path)
        _obs_keys = set(model.observation_space.spaces.keys())

        obs, _        = env.reset()
        if self.state_path:
            with open(self.state_path, "rb") as f:
                env._pyboy.load_state(f)
            obs, _, _, _, _ = env.step(0)

        lstm_states   = None
        episode_start = True
        ep_reward     = 0.0
        ep_steps      = 0
        episode       = 1

        while self.alive:
            try:
                action, lstm_states = model.predict(
                    {k: v for k, v in obs.items() if k in _obs_keys},
                    state         = lstm_states,
                    episode_start = episode_start,
                    deterministic = False,
                )
                obs, reward, terminated, truncated, info = env.step(int(action))
            except Exception:
                break

            episode_start = terminated or truncated
            ep_reward    += float(reward)
            ep_steps     += 1

            state = info.get("state", {})
            bd    = info.get("reward_breakdown", {})
            party = state.get("party", [])

            # Capture Game Boy screen as RGB numpy array
            try:
                pil_img     = env._pyboy.screen.image.convert("RGB")
                screen_arr  = np.array(pil_img, dtype=np.uint8)   # (144, 160, 3)
            except Exception:
                screen_arr  = None

            with self._lock:
                self.map_id           = state.get("map_id",   self.map_id)
                self.x                = state.get("player_x", self.x)
                self.y                = state.get("player_y", self.y)
                self.last_action      = int(action)
                self.ep_reward        = ep_reward
                self.ep_steps         = ep_steps
                self.reward_breakdown = dict(bd)
                self.badges           = bin(state.get("badges", 0)).count("1")
                self.lead_level       = party[0]["level"] if party else 0
                self.party_size       = len(party)
                self.in_battle        = bool(state.get("in_battle", False))
                self.episode          = episode
                self.screen_pixels    = screen_arr

            if terminated or truncated:
                ep_reward     = 0.0
                ep_steps      = 0
                episode      += 1
                lstm_states   = None
                episode_start = True
                obs, _        = env.reset()

        env.close()

    def snapshot(self) -> dict:
        """Thread-safe copy of this bot's current state."""
        with self._lock:
            return {
                "bot_id":           self.bot_id,
                "map_id":           self.map_id,
                "x":                self.x,
                "y":                self.y,
                "last_action":      self.last_action,
                "ep_reward":        self.ep_reward,
                "ep_steps":         self.ep_steps,
                "reward_breakdown": dict(self.reward_breakdown),
                "badges":           self.badges,
                "lead_level":       self.lead_level,
                "party_size":       self.party_size,
                "episode":          self.episode,
                "in_battle":        self.in_battle,
                "alive":            self.alive,
                "error":            self.error,
                "screen_pixels":    (
                    self.screen_pixels.copy()
                    if self.screen_pixels is not None else None
                ),
            }


# ── World-map geometry ────────────────────────────────────────────────────────

def _build_map_rects(scale: float, padding: int) -> tuple[
    dict[int, pygame.Rect], int, int
]:
    if not MAP_GLOBAL_ORIGIN:
        return {}, 400, 400

    min_gx = min(gx for gx, _ in MAP_GLOBAL_ORIGIN.values())
    min_gy = min(gy for _, gy in MAP_GLOBAL_ORIGIN.values())
    max_gx = max(gx + _MAP_WIDTH.get(mid, MAP_SIZE.get(mid, 10)) for mid, (gx, _) in MAP_GLOBAL_ORIGIN.items())
    max_gy = max(gy + _MAP_HEIGHT.get(mid, 10)     for mid, (_, gy) in MAP_GLOBAL_ORIGIN.items())

    world_w = int(round((max_gx - min_gx) * scale + padding * 2))
    world_h = int(round((max_gy - min_gy) * scale + padding * 2))

    rects: dict[int, pygame.Rect] = {}
    for mid, (gx, gy) in MAP_GLOBAL_ORIGIN.items():
        px = int(round((gx - min_gx) * scale + padding))
        py = int(round((gy - min_gy) * scale + padding))
        pw = max(1, int(round(_MAP_WIDTH.get(mid, MAP_SIZE.get(mid, 10)) * scale)))
        ph = max(1, int(round(_MAP_HEIGHT.get(mid, 10) * scale)))
        rects[mid] = pygame.Rect(px, py, pw, ph)

    return rects, world_w, world_h


def _world_to_screen(
    map_id: int, x: int, y: int,
    rects: dict[int, pygame.Rect],
    scale: float,
) -> tuple[int, int] | None:
    r = rects.get(map_id)
    if r is None:
        return None
    return (
        int(round(r.left + x * scale + scale / 2.0)),
        int(round(r.top + y * scale + scale / 2.0)),
    )


# ── Bot screen grid rendering ──────────────────────────────────────────────────

def _render_screens(
    surface: pygame.Surface,
    font_sm: pygame.font.Font,
    bots: list[BotThread],
    x_offset: int,
    panel_w: int,
    colors: list[tuple[int, int, int]],
    n_cols: int,
    screen_w: int,
    screen_h: int,
) -> None:
    """Draw a grid of individual bot Game Boy screens on the right panel."""
    surface.fill((20, 20, 30), rect=pygame.Rect(x_offset, 0, panel_w, surface.get_height()))

    cell_w = screen_w + _PADDING
    cell_h = screen_h + _STAT_H + _PADDING

    for bot in bots:
        s   = bot.snapshot()
        idx = s["bot_id"]
        col = colors[idx % len(colors)]

        col_idx = idx % n_cols
        row_idx = idx // n_cols
        cx = x_offset + col_idx * cell_w + _PADDING // 2
        cy =            row_idx * cell_h + _PADDING // 2

        # ── Draw Game Boy screen ───────────────────────────────────────────
        if s["screen_pixels"] is not None:
            # pygame surfarray expects (W, H, 3) — transpose from (H, W, 3)
            arr  = s["screen_pixels"].swapaxes(0, 1)
            surf = pygame.surfarray.make_surface(arr)
            surf = pygame.transform.scale(surf, (screen_w, screen_h))
        else:
            # placeholder while bot is starting up
            surf = pygame.Surface((screen_w, screen_h))
            surf.fill((40, 40, 40))
            wait_txt = font_sm.render("Starting…", True, (120, 120, 120))
            surf.blit(wait_txt, (4, 4))

        # Battle: draw a colored border around the screen
        border_col = (255, 220, 0) if s["in_battle"] else col
        pygame.draw.rect(surface, border_col,
                         pygame.Rect(cx - 2, cy - 2, screen_w + 4, screen_h + 4), 2)
        surface.blit(surf, (cx, cy))

        # ── Bot number tag (top-left corner of screen) ─────────────────────
        tag = font_sm.render(f" {idx + 1} ", True, (0, 0, 0))
        tag_bg = pygame.Surface((tag.get_width(), tag.get_height()))
        tag_bg.fill(col)
        surface.blit(tag_bg, (cx, cy))
        surface.blit(tag,    (cx, cy))

        # ── Stat bar below screen ──────────────────────────────────────────
        sy = cy + screen_h + 2
        
        map_name   = MAP_NAMES.get(s["map_id"], f"M{s['map_id']}")
        # Shorten long map names
        map_name = map_name.replace("Route", "R").replace("Viridian", "Virid").replace("Pokemon", "P")[:12]
        action_lbl = _ACTION_LABEL[s["last_action"]] if s["last_action"] < len(_ACTION_LABEL) else "?"
        
        # Line 1: compact stats
        line1 = f"{s['episode']}  r{s['ep_reward']:+.0f}  b{s['badges']}  L{s['lead_level']}({s['party_size']})"
        t1 = font_sm.render(line1, True, col)
        surface.blit(t1, (cx, sy))
        
        # Line 2: map + action
        line2 = f"{map_name} [{action_lbl}]"
        t2_map = font_sm.render(line2, True, (180, 180, 180))
        surface.blit(t2_map, (cx, sy + t1.get_height()))

        # Line 3: reward events (only non-quiet non-zero)
        bd     = s["reward_breakdown"]
        events = [
            f"{'+'if v>0 else ''}{v:.1f}{k[:4]}"
            for k, v in bd.items()
            if k not in _QUIET and abs(v) > 0.001
        ]
        if events:
            # Limit to first 5 events to prevent overflow
            events_str = "  ".join(events[:5])
            t3 = font_sm.render(events_str, True, (190, 190, 190))
            surface.blit(t3, (cx, sy + t1.get_height() + t2_map.get_height()))

        # Error (line 4 if present)
        if s["error"]:
            err = font_sm.render(f"ERR:{s['error'][:25]}", True, (255, 80, 80))
            surface.blit(err, (cx, sy + t1.get_height() + t2_map.get_height() + (t3.get_height() if events else 0)))


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    model_path: str,
    rom_path:   str           = "Pokemon_Red.gb",
    n_bots:     int           = 4,
    speed:      int           = 0,
    state_path: str | None    = None,
    max_steps:  int           = 8_192,
) -> None:
    if not Path(model_path).exists():
        sys.exit(f"[multiplay] Model not found: {model_path}")
    if not Path(rom_path).exists():
        sys.exit(f"[multiplay] ROM not found: {rom_path}")

    colors = _BOT_COLORS

    # ── Fixed window + panel geometry ──────────────────────────────────────
    win_w = _WIN_W
    win_h = _WIN_H
    map_panel_w = int(round(win_w * _MAP_FRAC))
    screens_w = win_w - map_panel_w

    # ── Compute bot-screen geometry to fit fixed right panel ──────────────
    n_cols     = max(1, math.ceil(math.sqrt(n_bots)))
    n_rows     = math.ceil(n_bots / n_cols)

    avail_screens_w = max(80, screens_w - (n_cols + 1) * _PADDING)
    avail_screens_h = max(80, win_h - (n_rows + 1) * _PADDING - n_rows * _STAT_H)
    screen_scale = min(
        1.0,
        avail_screens_w / (_SCREEN_NATIVE_W * n_cols),
        avail_screens_h / (_SCREEN_NATIVE_H * n_rows),
    )
    screen_scale = max(0.20, screen_scale)
    screen_w = max(1, int(round(_SCREEN_NATIVE_W * screen_scale)))
    screen_h = max(1, int(round(_SCREEN_NATIVE_H * screen_scale)))

    # ── Compute world-map geometry and fit it inside left panel ────────────
    _, base_w, base_h = _build_map_rects(_SCALE, 0)
    map_scale = _SCALE
    if base_w > 0 and base_h > 0:
        target_w = max(1, map_panel_w - 2 * _PADDING)
        target_h = max(1, win_h - 2 * _PADDING)
        fit = min(target_w / base_w, target_h / base_h)
        map_scale = max(0.10, _SCALE * fit)

    rects, world_w, world_h = _build_map_rects(map_scale, 0)
    map_x = max(0, (map_panel_w - world_w) // 2)
    map_y = max(0, (win_h - world_h) // 2)
    rects = {mid: r.move(map_x, map_y) for mid, r in rects.items()}
    tile_px = max(1, int(math.ceil(map_scale)))

    # ── Start bots ────────────────────────────────────────────────────────
    print(f"[multiplay] Starting {n_bots} bots (speed={speed})  "
          f"grid={n_cols}×{n_rows}  screen={screen_w}×{screen_h}...")
    bots: list[BotThread] = []
    for i in range(n_bots):
        bt = BotThread(i, model_path, rom_path, speed, state_path, max_steps)
        bt.start()
        bots.append(bt)
        time.sleep(0.3)

    visited_tiles: dict[int, set[tuple[int, int]]] = {}

    # ── Pygame setup ──────────────────────────────────────────────────────
    pygame.init()
    pygame.display.set_caption(f"Pokemon Red — {n_bots} bots")
    screen = pygame.display.set_mode((win_w, win_h))
    clock  = pygame.time.Clock()

    try:
        font_sm = pygame.font.SysFont("Consolas", 11)
    except Exception:
        font_sm = pygame.font.Font(None, 13)

    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        snapshots = [bot.snapshot() for bot in bots]
        for s in snapshots:
            mid = int(s["map_id"])
            tx = int(s["x"])
            ty = int(s["y"])
            w = _MAP_WIDTH.get(mid, MAP_SIZE.get(mid, 10))
            h = _MAP_HEIGHT.get(mid, 10)
            if 0 <= tx < w and 0 <= ty < h:
                visited_tiles.setdefault(mid, set()).add((tx, ty))

        # ── World map (left panel) ─────────────────────────────────────────
        screen.fill((12, 12, 20))
        screen.fill((15, 15, 25), rect=pygame.Rect(0, 0, map_panel_w, win_h))

        for mid, rect in rects.items():
            fill = (_DEFAULT_MAP_FILL_PC_DARK if mid in POKECENTER_MAPS
                    else _MAP_FILL_DARK.get(mid, _DEFAULT_MAP_FILL_DARK))
            pygame.draw.rect(screen, fill, rect)
            pygame.draw.rect(screen, (80, 80, 100), rect, 1)
            label = font_sm.render(MAP_NAMES.get(mid, f"M{mid}"), True, (160, 160, 180))
            screen.blit(label, (rect.left + 2, rect.top + 2))

        for mid, tiles in visited_tiles.items():
            rect = rects.get(mid)
            if rect is None:
                continue
            fill = (_DEFAULT_MAP_FILL_PC if mid in POKECENTER_MAPS
                    else _MAP_FILL.get(mid, _DEFAULT_MAP_FILL))
            for tx, ty in tiles:
                px = rect.left + int(round(tx * map_scale))
                py = rect.top + int(round(ty * map_scale))
                tr = pygame.Rect(px, py, tile_px, tile_px).clip(rect)
                if tr.width > 0 and tr.height > 0:
                    pygame.draw.rect(screen, fill, tr)

        # ── Bot dots on world map ──────────────────────────────────────────
        for s in snapshots:
            pos = _world_to_screen(s["map_id"], s["x"], s["y"], rects, map_scale)
            if pos is None:
                continue
            col = colors[s["bot_id"] % len(colors)]
            if s["in_battle"]:
                pygame.draw.circle(screen, (255, 255, 100), pos, 7, 2)
            pygame.draw.circle(screen, col, pos, 4)
            lbl = font_sm.render(str(s["bot_id"] + 1), True, (240, 240, 240))
            screen.blit(lbl, (pos[0] + 5, pos[1] - 6))

        # ── Bot screens (right panel) ──────────────────────────────────────
        _render_screens(screen, font_sm, bots, map_panel_w, screens_w, colors, n_cols, screen_w, screen_h)

        pygame.display.flip()

        if all(not b.is_alive() for b in bots):
            print("[multiplay] All bots finished.")
            time.sleep(2)
            running = False

    # ── Shutdown ───────────────────────────────────────────────────────────
    for b in bots:
        b.alive = False
    pygame.quit()
    print("[multiplay] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch N bots play Pokemon Red simultaneously.")
    parser.add_argument("model",                    help="Path to trained RecurrentPPO .zip")
    parser.add_argument("--rom",    default="Pokemon_Red.gb")
    parser.add_argument("--n",      type=int, default=4,   help="Number of bots (default: 4)")
    parser.add_argument("--speed",  type=int, default=0,   help="Emulation speed 0=unlimited")
    parser.add_argument("--state",  default=None,          help="Optional .state file for all bots")
    parser.add_argument("--max-steps", type=int, default=8_192,
                        help="Episode length cap (default: 8192)")
    args = parser.parse_args()

    run(
        model_path = args.model,
        rom_path   = args.rom,
        n_bots     = args.n,
        speed      = args.speed,
        state_path = args.state,
        max_steps  = args.max_steps,
    )
