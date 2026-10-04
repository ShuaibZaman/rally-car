# 00 — Program

Reference: [docs/RALLY_AI_LAB_SPEC.md](../RALLY_AI_LAB_SPEC.md) §1, §34, §39, §40, §50.

This file is the index. Each child plan is 5–8 steps and names the spec sections it implements. An agent should be able to execute one child plan from that file plus the spec, without the original chat.

## Wave order

| Wave | Who | Plans | Done when |
| --- | --- | --- | --- |
| 0 | one agent | this index, spec, venv, package skeleton, experiment-id schema | imports resolve, `py -3.13` venv exists |
| 1 | three agents, disjoint files | 01 simulator, 02 tracks, 04 rewards | headless tests for momentum, seeds, reward sums |
| 2 | one integrator | 03 sensors and classical baseline | Gymnasium env steps; pure pursuit can finish an easy lap |
| 3 | one agent | 05 PPO lab | short PPO smoke writes one checkpoint bundle and metrics JSON |
| 4 | two agents | 06 harness, 07 presentation | configs and eval script exist; dashboard modes render |
| 5 | one agent | 08 extensions | scaffolds import; nothing in this wave is trained |

Do not start wave 2 until `environment/types.py` exists. Do not start wave 3 until the env and reward variants exist.

## File ownership

No two agents edit the same module in the same wave.

- Simulator agent: `environment/physics.py`, `environment/car.py`, `tests/test_physics.py`
- Track agent: `environment/track.py`, `environment/generator.py`, `tests/test_track.py`
- Reward agent: `environment/rewards.py`, `tests/test_rewards.py`, `training/configs/rewards/`
- Integrator: `environment/sensors.py`, `environment/env.py`, `agents/classical.py`, `tests/test_env.py`
- Training agent: `training/train.py`, `training/metrics.py`, `training/evaluate.py`, `visualization/`
- Extension agent: `environment/surfaces.py`, `environment/curriculum.py`, `environment/weather.py`, `environment/obstacles.py`, `agents/ppo.py`, `agents/behavior_cloning.py`, `agents/policies/`, `visualization/web/`

Shared contracts live in `environment/types.py` and `training/experiment.py`. Change those only in wave 0, or in a single follow-up, never from two agents at once.

## Contracts

State: `x`, `y`, `theta`, `vx`, `vy`, `omega`.

Action: `steering` in [-1, 1], `throttle` in [0, 1], `brake` in [0, 1]. Separate brake. No reverse in the default physics.

Track handle: generated from an integer seed plus a preset name (`easy`, `medium`, `hard`, `extreme`). Same seed and preset must regenerate the same centerline, width, and checkpoints.

Observation, reward variant names (`A`–`D`), and experiment ids are defined in plans 03, 04, and the experiment schema. They stay stable once wave 0 lands.

## Smoke-only training rule

Implementation work runs unit tests and at most one short PPO smoke (`--max-steps` small enough to finish in minutes) on one easy seeded track.

Do not launch architecture sweeps, multi-seed studies, SAC/TD3 training, vision training, or overnight jobs unless the user explicitly asks. `evaluate.py` must be able to aggregate mean and spread across seeds, and that path stays unrun for a matrix.

## Interpreter

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Never build this venv with system Python 3.14. CUDA wheels come from the PyTorch index, not default PyPI.

## Child plans

1. [01-simulator.md](01-simulator.md) — §3, §4, §50 phase 1
2. [02-tracks.md](02-tracks.md) — §7, §8
3. [03-sensors.md](03-sensors.md) — §5, §21, §30
4. [04-rewards.md](04-rewards.md) — §9, §10, §11, §27
5. [05-ppo-lab.md](05-ppo-lab.md) — §12, §23, §25, §26, §35–§37, §51
6. [06-experiments.md](06-experiments.md) — §13–§15, §33, §34, §38, §48
7. [07-presentation.md](07-presentation.md) — §19, §20, §22, §24, §42–§47
8. [08-extensions.md](08-extensions.md) — §6, §16–§18, §28–§32, §40–§41
