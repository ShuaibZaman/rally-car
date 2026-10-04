# 05 — PPO lab

Spec: §12, §23, §25, §26, §35, §36, §37, §50 phases 3 and 5, §51.

One short Stable-Baselines3 PPO run proves the loop. It is not a claim that the car learned to race.

## Steps

1. `training/train.py` builds the env from an experiment config, wraps it with Monitor, and runs PPO with MLP hidden sizes `[128, 128]`, Adam, learning rate `3e-4`.
2. Default smoke: one easy track seed, a few thousand environment steps, one update at least. Write `training/checkpoints/ppo_smoke.zip` plus a JSON bundle: experiment id, track seed, env config, model config, training config, metrics, episode reward history, and any logged loss values.
3. Metrics module (§25, §26): completion, lap time, progress, collisions, average speed, off-track fraction, centerline deviation, steering smoothness, throttle efficiency, recoveries, distance, fuel as integrated throttle. Rally score is a weighted combination and is never returned without the components.
4. Four-panel pygame lab (§2, §35): live track with rays and trajectory; model / episode / reward / lap / progress; reward and loss sparklines from the bundle; steering, throttle, and brake bars.
5. Sparkline data comes from the callback and the logger. If a smoke run has only a few episodes, show those points rather than a fabricated curve (§36).
6. Tests cover the score formula and that a tiny rollout fills every metric field. The GPU smoke itself is a manual command, not part of unit tests.

## Command

```powershell
.\.venv\Scripts\python.exe -m training.train --max-steps 4096 --preset easy --track-seed 1000000 --reward C
```

Do not raise `--max-steps` into a full training job from this plan.
