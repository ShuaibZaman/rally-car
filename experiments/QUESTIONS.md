# Research questions

Source: spec §48. The harness in `training/catalog.py` and `training/evaluate.py` can host any of these. The first three are the ones this lab is set up to answer before a full sweep is launched. None of them are answered by the smoke checkpoint.

## First

1. **Reward shaping.** How does reward shaping influence the tradeoff between lap completion, speed, and safety?
   Configs: `reward_A`, `reward_B`, `reward_C`, `reward_D`. Same PPO MLP. Variants are defined in `environment/rewards.py`. A is the progress-only hacking fixture.

2. **Architecture at fixed capacity.** How does network depth affect driving performance when trunk parameter count stays roughly constant?
   Configs: `depth2`, `depth4`, `depth6`. A unit test checks the trunks are within 20 percent of each other.

3. **Unseen-track generalization.** Does a policy trained on the train seed pool drive tracks from the disjoint test pool?
   Pools: `environment/generator.py` (`split_seeds`). Eval entry: `training/evaluate.py --split test`. Config: `generalization_holdout`.

## Documented, not scheduled

4. **RL algorithm.** How do PPO, SAC, and TD3 differ in learning speed, stability, and final driving performance under a matched hidden size?
   Configs: `algo_ppo`, `algo_sac`, `algo_td3`. Builders exist. They are not trained in the smoke run.

5. **Representation.** How does learning from raw pixels compare with engineered raycasts?
   Config: `vision_cnn`. `training/train.py` refuses `--observation vision` unless `--allow-vision` is passed.

6. **Curriculum.** Does progressive difficulty reduce the interactions needed for competent driving?
   Stages: `environment/curriculum.py`. The trainer does not auto-advance.

7. **Memory.** Does a recurrent architecture help under partial observability?
   Config: `memory_lstm` (`sb3-contrib` `RecurrentPPO`). Not part of the smoke run.

Report mean and spread across seeds when a real comparison is run (`training/metrics.py` `aggregate_metrics`). Do not treat one smoke seed as that study.
