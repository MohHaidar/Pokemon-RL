"""
debug_memory.py – Memory diff tool for finding correct RAM addresses.

Usage:
    uv run python debug_memory.py

Controls in game window: Arrow keys | Z=A | X=B | Enter=Start | Bksp=Select

Snapshot trigger: press S in the GAME WINDOW (mapped to Select by PyBoy).
  - First S press  → saves BEFORE snapshot, prints "BEFORE saved"
  - Second S press → diffs against BEFORE, prints changed addresses
  - Each subsequent S press → diffs against previous snapshot

Everything runs on the main thread so SDL2 stays responsive.
"""

from __future__ import annotations
import os
os.environ.setdefault("SDL_RENDER_DRIVER", "software")

from pyboy import PyBoy
from pyboy.utils import WindowEvent

STATE_PATH = "initial_state.state"

KNOWN = {
    0xC3A0: "ADDR_TEXT_BOX (current guess)",
    0xCF0D: "ADDR_NPC_TALK_FLAG (current guess)",
    0xD057: "ADDR_IN_BATTLE",
    0xD35E: "ADDR_MAP_ID",
    0xD361: "ADDR_PLAYER_Y",
    0xD362: "ADDR_PLAYER_X",
    0xCC3B: "wTextBoxID",
    0xCC26: "wCurrentMenuItem",
    0xCC24: "wTextDest",
    0xD730: "wGameProgressFlags",
    0xCF13: "wSpriteStateData2",
    0xCF0B: "wSpriteStateData near",
}

# Only scan WRAM (C000-DFFF) — fast enough per frame
WRAM_START = 0xC000
WRAM_END   = 0xDFFF


def snapshot(m) -> bytes:
    return bytes(m[a] for a in range(WRAM_START, WRAM_END + 1))


def diff(before: bytes, after: bytes) -> list[tuple[int, int, int]]:
    changes = []
    for i, (b, a) in enumerate(zip(before, after)):
        if b != a:
            changes.append((WRAM_START + i, b, a))
    return changes


def print_diff(changes: list) -> None:
    if not changes:
        print("  (no changes)")
        return
    print(f"  {len(changes)} addresses changed:")
    for addr, old, new in changes:
        label = KNOWN.get(addr, "")
        tag   = f"  ← {label}" if label else ""
        print(f"    0x{addr:04X}  {old:3d}→{new:3d}  (0x{old:02X}→0x{new:02X}){tag}")


def main() -> None:
    print("=" * 60)
    print("Pokemon Red – Memory Diff Debugger")
    print("=" * 60)
    print("Press BACKSPACE (Select) in the game window to take snapshots.")
    print("  1st press → BEFORE snapshot")
    print("  2nd press → diff printed")
    print("  Each press after → new diff from last snapshot")
    print("Close the game window to quit.")
    print("=" * 60)

    pyboy = PyBoy("Pokemon_Red.gb", window="SDL2")
    pyboy.set_emulation_speed(1)
    with open(STATE_PATH, "rb") as f:
        pyboy.load_state(f)

    snap_before: bytes | None = None
    snap_n = 0
    select_was_down = False

    print("\n[running] Switch to game window. Press Backspace (Select) to snapshot.\n")

    while pyboy.tick(1, True):
        m = pyboy.memory

        # Detect Select button press (rising edge) via memory register FF00 area
        # PyBoy exposes button state; we poll the joypad RAM at 0xFF00
        # Select is bit 2 of the low nibble when P14 is selected (0xFF00 & 0x08)
        # Simpler: watch for the WindowEvent — use memory poll instead
        joypad = m[0xFFB4]  # wCurButtonPressed — set for buttons pressed THIS frame
        select_down = bool(joypad & 0x04)  # bit 2 = Select

        if select_down and not select_was_down:
            snap = snapshot(m)
            snap_n += 1
            if snap_before is None:
                snap_before = snap
                print(f"[snap {snap_n}] BEFORE saved. Now do ONE action, then press Select again.\n")
            else:
                changes = diff(snap_before, snap)
                print(f"\n{'='*55}")
                print(f"[snap {snap_n}] DIFF")
                print(f"{'='*55}")
                print_diff(changes)
                print(f"{'='*55}\n")
                snap_before = snap
                print("New BEFORE saved. Do next action, then press Select.\n")

        select_was_down = select_down

    pyboy.stop()
    print("Done.")


if __name__ == "__main__":
    main()
