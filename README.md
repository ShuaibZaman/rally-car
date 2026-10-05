# Autonomous Rally AI Lab

A 2D top-down rally car on seeded tracks, built so reinforcement-learning choices are visible. The question is not only whether a network can drive. It is how the algorithm, the reward, and the network change completion, lap time, and crashes.

The canonical brief is [docs/RALLY_AI_LAB_SPEC.md](docs/RALLY_AI_LAB_SPEC.md). The build is split into [docs/plans/](docs/plans/).

## Layout

```
environment/     physics, tracks, sensors, rewards, gymnasium env
agents/          classical driver, policy modules, PPO loss, SAC/TD3 builders
training/        smoke trainer, evaluator, experiment configs
experiments/     the research questions and what is actually scheduled
visualization/   pygame lab and matplotlib figures
```

## Setup

Run everything from the project root (`rally-car`). Use PowerShell.

Python 3.13. Install torch 2.10 from the PyTorch CUDA index first. Torch 2.11+cu128 fails to import on this interpreter (a JIT parser error in `torch/nn/modules/rnn.py`). Then install the rest. Do not use system Python 3.14 for this environment.

```powershell
cd path\to\rally-car
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Race track window (pygame lab)

The only command that opens the on-screen track is `visualization.dashboard`. It starts a pygame window titled **Autonomous Rally AI** (four panels: live track, model info, training sparkline, actions).

Start it from the project root with the venv Python:

```powershell
cd path\to\rally-car
.\.venv\Scripts\python.exe -m visualization.dashboard --mode <mode>
```

If you omit `--mode`, the default is `classical` (pure-pursuit autopilot on the easy fixture track).

| Mode | What you see |
|------|----------------|
| `manual` | You drive with the keyboard. |
| `classical` | Pure-pursuit baseline, no neural network. |
| `policy` | Loaded PPO checkpoint (`--checkpoint` or newest in `--checkpoint-dir`). |
| `ghosts` | Classical, random, and policy on the same track at once. |
| `evolution` | Six small views (random → early → late checkpoints). |
| `compare` | Bar chart from `data/results/eval_*.json` (no live car). |

Common examples:

```powershell
# Drive yourself (arrows or WASD; R resets the car)
.\.venv\Scripts\python.exe -m visualization.dashboard --mode manual

# Watch the classical baseline; build a crash heatmap over laps
.\.venv\Scripts\python.exe -m visualization.dashboard --mode classical --until-lap --heatmap

# Watch a trained policy (after smoke train, or your own checkpoint)
.\.venv\Scripts\python.exe -m visualization.dashboard --mode policy --checkpoint training\checkpoints\ppo_smoke.zip
```

Useful flags: `--track-seed`, `--preset` (`easy` / `hard`), `--reward` (`A`–`D`), `--follow`, `--replay <trajectory.json>`. In the window, **1–6** switch modes, **F** follow camera, **L** toggle lidar rays, **P** pause and scrub with arrow keys, **[** **]** cycle checkpoints in policy mode, **E** export a figure.

### Close the race track window

The dashboard runs until you stop it or it hits its step limit (`--steps`, default 2400 simulation steps).

1. **Esc** — quit immediately (recommended).
2. **Window close (X)** — same as Esc.
3. **Ctrl+C** in the PowerShell window — stops the process if the window is stuck or focus is lost.

When the process exits, pygame shuts down and your terminal prompt returns. Training and evaluation commands below do **not** open this window.

## Smoke train

A few thousand PPO steps on one easy track. This checks the loop and writes a checkpoint. It is not a finished racing policy. No pygame window.

```powershell
.\.venv\Scripts\python.exe -m training.train --max-steps 4096 --preset easy --track-seed 1000000 --reward C
.\.venv\Scripts\python.exe -m training.evaluate --checkpoint training\checkpoints\ppo_smoke.zip --split test --seed-limit 1 --episode-steps 1500
```

To watch that checkpoint on the track, use `--mode policy` as shown above.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Full algorithm sweeps, multi-seed studies, vision training, and overnight runs stay off until you ask for them. See [experiments/QUESTIONS.md](experiments/QUESTIONS.md).
