# Algorithm experiments

Spec §12, §34.

PPO is the baseline (`training/train.py`, Stable-Baselines3). SAC and TD3 builders are in `agents/sac.py` and `agents/td3.py`. The custom clipped objective in `agents/ppo.py` is for loss-level checks; it is not a second training stack.

Configs `algo_ppo`, `algo_sac`, and `algo_td3` share hidden size `[128, 128]` and reward C. Running them as a bake-off is a separate request from the smoke checkpoint.
