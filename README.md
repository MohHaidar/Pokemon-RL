# Pokemon Red — Reinforcement Learning

A reinforcement learning experiment that trains an AI agent to play **Pokemon Red** using [PyBoy](https://github.com/Bonsai88/PyBoy) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (RecurrentPPO).

The agent learns from screen pixels, a structured game-state vector, and a visited-tile minimap, reading RAM directly to build a rich reward signal.

---

## How it works

| Component | Details |
|---|---|
| **Emulator** | PyBoy 2.x (Game Boy emulator in Python) |
| **Algorithm** | RecurrentPPO (PPO with LSTM memory) |
| **Policy** | `MultiInputLstmPolicy` — CNN + state vector + minimap → LSTM(256) |
| **Observation** | Dict: 4 stacked 84×84 grayscale frames + 91-dim game-state vector + 21×21 minimap |
| **Actions** | 9 — Noop / Up / Down / Left / Right / A / B / Start / Select |
| **Episode end** | Max 8 192 steps, score floor (−500), or stuck timeout (500 steps without a new tile) |

### Model architecture

```
screen  (84×84×4)  → Conv(32,8×8,s4) → Conv(64,4×4,s2) → Conv(64,3×3,s1) → Linear(256) ─┐
state   (91-dim)   → Linear(64) ──────────────────────────────────────────────────────────┼→ 384-dim → LSTM(256) → Actor / Critic
minimap (21×21×1)  → Conv(16,3×3,s1) → Conv(32,3×3,s1) → Linear(64) ──────────────────────┘
```

- **Screen CNN** — classic DQN/Atari 3-layer architecture, outputs 256-dim.
- **State linear** — encodes HP, XP, levels, badges, position, battle stats, and more into 64-dim.
- **Minimap CNN** — 21×21 binary grid of visited tiles centred on the player, outputs 64-dim.

### State vector (91-dim)

| Indices | Content |
|---|---|
| `[0]` | Lead HP ratio |
| `[1]` | Lead level / 100 |
| `[2]` | Lead status multiplier (1.0 = healthy) |
| `[3–7]` | Bench slots 2–6 HP ratios |
| `[8–12]` | Bench slots 2–6 levels / 100 |
| `[13]` | Party size / 6 |
| `[14]` | Badges / 8 |
| `[15–16]` | Player x, y (normalised to map size) |
| `[17–35]` | Misc exploration / battle scalars (in-battle flag, enemy HP ratio, etc.) |
| `[36–56]` | HMs/items flags, bag count, battle info, moves, PP, dialogue flag |
| `[57–62]` | Per-slot survival probability (ttd-based estimate) |
| `[63–68]` | Per-slot turns-to-die (ttd) vs current enemy |
| `[69–74]` | Per-slot turns-to-kill (ttk) vs current enemy |
| `[75–80]` | Per-slot speed advantage vs enemy |
| `[81–84]` | Player stat stages Atk/Def/Spd/Spc (0 = neutral) |
| `[85–88]` | Enemy stat stages Atk/Def/Spd/Spc (0 = neutral) |
| `[89]` | Threat ratio (enemy damage / lead HP) |
| `[90]` | Catchability estimate [0, 1] |

### Reward system

Rewards are scaled by two phase multipliers that shift focus from exploration to combat as badges are earned:
- `explore_mult`: 1.0 → 0.2 (exploration rewards shrink as badges are earned)
- `battle_mult`:  1.0 → 2.5 (combat rewards grow as badges are earned)

#### Exploration

| Event | Reward |
|---|---|
| Step onto a new tile | `+1.3 × explore_mult` |
| Enter a previously unseen map | `+8.0 × explore_mult` |
| First visit to a milestone map | `+10–20` (one-time, see below) |
| New northward distance record (Pallet + Route 1 only) | `+1.0 × explore_mult` per tile |
| Walk into a wall | `−0.05` |
| Revisit tile | `−0.12` |
| No new tile for many steps (stale) | up to `−0.12` per step |
| Rapid door spam (< 60 steps between map changes) | `−3.0` |

**Milestone maps** (one-time rewards, suppressed on death-respawn):

| Map | Reward |
|---|---|
| Viridian City (map 1) | `+15.0` |
| Route 22 (map 33) | `+10.0` |
| Viridian Forest S Gate (map 50) | `+15.0` |
| Viridian Forest (map 51) | `+20.0` |

#### Combat

| Event | Reward |
|---|---|
| Deal damage to enemy | `+0.7 × HP dealt × battle_mult` |
| Win a battle | `+15.0 × battle_mult` |
| Lower an enemy stat stage | small bonus × `battle_mult` |
| Throw a Poké Ball | `+15.0` |
| Catch a new species | `+100.0` |
| First visit to a gym map | `+20.0` |
| Earn a badge | `+1 000.0` |
| Pokémon levels up | `+120.0` |
| XP gained | `+xp_delta × 0.07` |
| Run from wild battle (low ratio) | `+5.0` |
| Run away when strong | `−15.0` |
| Lead faints / blackout | `−80.0` |

#### Healing / Pokémon Center

| Event | Reward |
|---|---|
| Enter a PC with HP < 25% | `+1.0` (once per visit) |
| Proximity to Nurse Joy when HP is low | up to `+0.40` per step |
| Full heal from Nurse Joy | `+200.0` (one-shot) |
| Per-step low-HP penalty (HP < 40%) | `−0.2` |

> **Faint-cycle math:** `DEATH_PENALTY (−80) + NEW_MAP (+8) + Viridian milestone (+15) = −57` net. Intentional deaths are always negative after the one-time milestone is consumed, and the milestone is suppressed on the respawn map if the bot died to trigger it.

---

## Project structure

```
pokemon RL/
├── Pokemon_Red.gb          # ROM (not included)
├── initial_state.state     # Starting save state (created by setup_state.py)
├── env.py                  # Gymnasium environment (PokemonRedEnv)
├── train.py                # RecurrentPPO training script
├── play.py                 # Watch a trained model play (single window)
├── multiplay.py            # Watch multiple bots in a grid (parallel display)
├── record.py               # Record human gameplay for behavioural cloning
├── bc_pretrain.py          # Pre-train the policy from recorded data (NLL loss)
├── setup_state.py          # One-time interactive setup to create the save state
├── ram_map.py              # Single source of truth for all RAM addresses + map data
├── game_helpers.py         # RAM read layer + pure query functions for reward/obs
├── tests/
│   ├── test_env.py         # Env constants, shapes, structural invariants
│   ├── test_game_helpers.py # Query-layer unit tests (no ROM needed)
│   └── test_ram_map.py     # Stage mult, catch probability, status helpers
├── runs/
│   ├── checkpoints/        # Checkpoints saved every 50 k steps
│   ├── best/               # Best checkpoint by mean reward
│   └── pokemon_ppo_final.zip
└── logs/                   # TensorBoard logs
```

---

## Setup

### Requirements

- Python 3.12 (managed by [uv](https://github.com/astral-sh/uv))
- A legal copy of the **Pokemon Red** ROM (`Pokemon_Red.gb`)

### Install dependencies

```bash
uv sync --all-groups
```

### Create the starting save state (run once)

```bash
uv run python setup_state.py
```

A game window opens. Play through the intro, name your character, pick your starter, and walk out into Pallet Town. Close the window to save `initial_state.state`.

> **Controls while playing:**
> | Key | Button |
> |---|---|
> | Arrow keys | D-pad |
> | Z | A |
> | X | B |
> | Enter | Start |
> | Backspace | Select |

---

## Training

```bash
uv run python train.py
```

| Flag | Default | Description |
|---|---|---|
| `--envs` | `4` | Parallel training environments |
| `--steps` | `10 000 000` | Total training timesteps |
| `--resume` | — | Path to a checkpoint `.zip` to continue from |
| `--seed` | `42` | Random seed |
| `--threshold` | `0.0` | Only save a "best" checkpoint when mean reward exceeds this |

**Examples**

```bash
# Start fresh with 6 parallel envs
uv run python train.py --envs 6

# Resume from a checkpoint
uv run python train.py --resume runs/checkpoints/pokemon_ppo_500000_steps

# Train for 50 M steps
uv run python train.py --steps 50_000_000
```

> **Scaling envs on resume:** batch size auto-scales with `--envs`.  
> Formula: `batch_size = round_power2(max(256, envs × 2048 // 48))`.  
> 6 envs → 256, 12 envs → 512, 24 envs → 1024. You can change `--envs` freely on resume.

### Monitor with TensorBoard

```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006`.

---

## Pre-training with behavioural cloning (optional)

Record human gameplay, then pre-train the policy before PPO to give the agent a head-start.

### Step 1 — Record gameplay

```bash
uv run python record.py --output recordings/my_run.pkl
```

An SDL2 window opens. Play normally; every step is saved. Press `Ctrl+C` to stop.

### Step 2 — Pre-train

```bash
uv run python bc_pretrain.py recordings/ --epochs 10 --output runs/bc_pretrain.zip
```

This trains the policy (same architecture as `train.py`) using NLL loss against your actions. The resulting `.zip` is a valid RecurrentPPO checkpoint.

### Step 3 — Fine-tune with PPO

```bash
uv run python train.py --resume runs/bc_pretrain.zip --envs 6
```

---

## Watch the agent play

### Single window

```bash
uv run python play.py runs/pokemon_ppo_final.zip
```

| Flag | Default | Description |
|---|---|---|
| `--speed` | `0` | Emulation speed (0 = unlimited, 1 = real-time, 2 = 2×) |
| `--state` | — | Optional save state to load before playing |
| `--rom` | `Pokemon_Red.gb` | Path to the ROM |

### Multi-bot grid

```bash
uv run python multiplay.py runs/pokemon_ppo_final.zip --envs 6
```

Displays all bots in a grid with per-bot reward, position, HP, and battle stats overlaid on each screen.

---

## Run tests

```bash
uv run python -m pytest tests/ -v
```

All tests are ROM-free (mock state dicts and constants only) and should pass in under a second.

---

## Tips

- **More envs = faster training**, but each one runs a full emulator. A good rule of thumb: `(CPU threads − 2)` envs to leave OS headroom.
- Training 10 M steps with 6 envs takes several hours on CPU. A GPU speeds up neural-net forward passes but the emulator is usually the bottleneck.
- The agent explores randomly early on. Purposeful navigation (buildings, Route 1, Viridian City) typically emerges after 1–3 M steps.
- Checkpoints are **incompatible** across observation-space or architecture changes. Start fresh after such changes.

