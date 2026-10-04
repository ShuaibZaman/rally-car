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

Python 3.13. Install torch 2.10 from the PyTorch CUDA index first. Torch 2.11+cu128 fails to import on this interpreter (a JIT parser error in `torch/nn/modules/rnn.py`). Then install the rest. Do not use system Python 3.14 for this environment.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Drive it

Keyboard (arrows or WASD, R resets, Esc quits):

```powershell
.\.venv\Scripts\python.exe -m visualization.dashboard --mode manual
```

Classical pure-pursuit baseline, no network:

```powershell
.\.venv\Scripts\python.exe -m visualization.dashboard --mode classical --until-lap --heatmap
```

## Smoke train

A few thousand PPO steps on one easy track. This checks the loop and writes a checkpoint. It is not a finished racing policy.

```powershell
.\.venv\Scripts\python.exe -m training.train --max-steps 4096 --preset easy --track-seed 1000000 --reward C
.\.venv\Scripts\python.exe -m training.evaluate --checkpoint training\checkpoints\ppo_smoke.zip --split test --seed-limit 1 --episode-steps 1500
```

`[` and `]` in policy mode move along `training/checkpoints/*.zip` when more than one checkpoint exists.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Full algorithm sweeps, multi-seed studies, vision training, and overnight runs stay off until you ask for them. See [experiments/QUESTIONS.md](experiments/QUESTIONS.md).
