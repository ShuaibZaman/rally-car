# 08 — Extensions

Spec: §6, §16, §17, §18, §28, §29, §31, §32, §40 stage 2, §41.

Scaffold the later lab. Do not train it in this plan.

## Steps

1. Surfaces: asphalt, gravel, mud, snow, dirt with grip and acceleration scales (§17). Weather: dry, rain, snow as further grip and drag multipliers (§18). Default remains dry asphalt.
2. Curriculum stages 1–7 mapping to presets and disturbance flags (§16). A helper returns the stage config. The trainer does not auto-advance.
3. Static obstacles sampled from a seed, and a moving-obstacle step (§28, §29). The env ignores them unless the flag is on.
4. Custom PPO loss (clipped surrogate, value loss, entropy) in `agents/ppo.py`, unit-tested on fake tensors. SAC and TD3 modules document that Stable-Baselines3 is the runner and expose a constructor hook. This is the §40 stage-2 hook, not a second training stack.
5. Behavior cloning: collect `(observation, expert action)` from the pure-pursuit driver; losses MSE, MAE, Huber; optimizer names Adam, AdamW, SGD (§31). A tiny supervised step on a short demo batch is enough. A function that copies BC weights into an SB3 policy is the §32 handoff and is not run as a long job.
6. Policy modules: MLP, residual MLP, CNN, CNN+LSTM (§6, §13, §15). Vision env mode renders an 84×84×3 top-down image and sets the observation space. `train.py` refuses vision unless `--allow-vision` is passed.
7. `visualization/web/README.md` describes the later React lab (§41) and points back at the pygame dashboard. No Node app in this plan.

## Tests

Surface table values, curriculum stage order, deterministic obstacles, PPO loss finite on a fake batch, BC loss names, vision observation shape. No GPU training.
