# Reward experiments

Spec §9, §10, §11, §27.

These files name **reward weights only**. Algorithm, architecture, optimizer, and learning rate are experiment fields, not reward fields.

| variant | objective | episode end |
| --- | --- | --- |
| A | progress only | does not end on collision or off-track (hacking fixture) |
| B | progress − collision | ends on collision |
| C | progress − collision − time − off-track, plus a lap bonus | ends on collision or a completed lap |
| D | C plus a racing-line term toward the inside of curves | same as C |

Per-step HUD data is `RewardBreakdown` in `environment/types.py`. The scalar reward is the sum of those terms.

JSON copies live in `training/configs/rewards/` after `environment.rewards.write_reward_configs` runs.
