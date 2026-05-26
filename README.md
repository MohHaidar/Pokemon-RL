# Pokemon Red — Reinforcement Learning

A reinforcement learning experiment that trains an AI agent to play **Pokemon Red** using [PyBoy](https://github.com/Bonsai88/PyBoy) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (RecurrentPPO).

The agent learns from screen pixels combined with a structured game-state vector, reading RAM addresses to build a rich reward signal.

---

## How it works

| Component | Details |
|---|---|
| **Emulator** | PyBoy 2.x (Game Boy emulator in Python) |
| **Algorithm** | RecurrentPPO (PPO with LSTM memory) |
| **Policy** | `MultiInputLstmPolicy` — CNN + state vector → LSTM(256) |
| **Observation** | Dict: 4 stacked 84×84 grayscale frames + 52-dim game-state vector |
| **Actions** | 8 — Up / Down / Left / Right / A / B / Start / Select |
| **Episode end** | Max steps reached, score floor (−500), or stuck timeout (512 steps) |

### Model architecture

```
screen (84×84×4) → Conv(32,8×8,s4) → Conv(64,4×4,s2) → Conv(64,3×3,s1) → Linear(256) ─┐
                                                                                           ├→ 320-dim → LSTM(256) → Actor / Critic
state  (52-dim)  → Linear(64) ─────────────────────────────────────────────────────────────┘
```

The CNN uses the classic DQN/Atari 3-layer architecture. The 52-dim state vector encodes HP, XP, level, badges, position, map ID, battle flags, party size, and more.

### Reward system

Rewards are scaled by two phase multipliers that shift focus from exploration to combat as badges are earned:
- `explore_mult`: 1.0 → 0.2 (exploration rewards shrink as more badges are earned)
- `battle_mult`: 1.0 → 2.5 (combat rewards grow as more badges are earned)

#### Exploration

| Event | Reward |
|---|---|
| Step onto a new tile | `+1.0 × explore_mult` |
| New personal-best distance from map entry point | `+3.0 × explore_mult` per unit |
| Enter a previously unseen map | `+5.0 × explore_mult` |
| Movement streak on new tiles (caps at 10) | `+streak × 0.2 × explore_mult` per step |
| Visual novelty — screen differs from bank by > 12% MAD | `+2.0 × explore_mult` |

The **visual novelty** system compares every frame (downsampled to 16×16 grayscale) against a bank of up to 2 000 previously seen scenes. Minor animations (NPC walking, water ripple, flowers) change < 5% of pixels and never trigger; entering a new area or room always does.

#### Combat

| Event | Reward |
|---|---|
| Deal damage to enemy | `+0.5 × HP dealt × battle_mult` |
| Win a battle | `+10.0 × battle_mult` |
| Catch a Pokémon | `+20.0` |
| Enter a gym (first time) | `+15.0` |
| Earn a badge | `+1 000.0` |
| Pokémon levels up | `+100.0` |
| XP gained | `+xp_delta × 0.05` |

#### Penalties

| Event | Penalty |
|---|---|
| Walk into a wall | `−0.05` per step |
| Revisit a recently seen tile (anti-loop) | `−0.1` |
| Idle in battle without acting | `−0.02` per step |
| Run from battle | `−15.0` |
| All Pokémon faint (blackout) | `−80.0` |
| Rapid door spam (< 30 steps between map changes) | `−3.0` |
| No new tile found in 60+ steps (stale exploration) | up to `−1.0` per step |

> **Faint-cycle math:** 3 wins × +10 = +30, faint = −80 → net −50. Farming battles until fainting is always net-negative.

---

## Project structure

```
pokemon RL/
├── Pokemon_Red.gb          # ROM (not included)
├── initial_state.state     # Starting save state created by setup_state.py
├── env.py                  # Gymnasium environment (PokemonRedEnv)
├── train.py                # RecurrentPPO training script
├── play.py                 # Watch a trained model play
├── setup_state.py          # One-time interactive setup to create the save state
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
| `--envs` | `4` | Parallel training environments (recommended: 6 for 8-thread CPU) |
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

### Monitor with TensorBoard

```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006` in your browser.

---

## Watch the agent play

```bash
uv run python play.py runs/pokemon_ppo_final.zip
```

| Flag | Default | Description |
|---|---|---|
| `--speed` | `0` | Emulation speed (0 = unlimited, 1 = real-time, 2 = 2×) |
| `--state` | — | Optional save state to load before playing |
| `--rom` | `Pokemon_Red.gb` | Path to the ROM |

**Examples**

```bash
# Watch at full speed
uv run python play.py runs/pokemon_ppo_final.zip

# Watch at real-time speed
uv run python play.py runs/pokemon_ppo_final.zip --speed 1

# Watch from a specific checkpoint
uv run python play.py runs/checkpoints/pokemon_ppo_1000000_steps.zip
```

The terminal prints a live stats line while it plays:

```
  Step     Reward  Level  Badges       HP  Map
    42      12.50      5       1    38/45   12
```

---

## Tips

- **More envs = faster training**, but each one runs a full emulator instance. A good rule of thumb is `(CPU threads − 2)` envs to leave headroom for the OS and training loop.
- Training 10 M steps with 4 envs takes several hours on CPU. A GPU speeds up the neural network forward passes but the bottleneck is the emulator.
- The agent will spend a long time exploring randomly at first. Meaningful behaviour (navigating, fighting, entering buildings) typically emerges after ~1–3 M steps.
- Save states let you start training from any point in the game — after getting all badges, after a specific event, etc.
- New checkpoints are incompatible with old ones whenever the observation space or model architecture changes. Always start fresh after such changes.
