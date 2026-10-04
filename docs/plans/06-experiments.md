# 06 — Experiment harness

Spec: §13, §14, §15, §33, §34, §38, §48.

Configs and an evaluator. Not a 360-run sweep.

## Steps

1. Experiment schema: algorithm, architecture name, hidden sizes, observation (`sensors` or `vision`), reward variant, optimizer, learning rate, train seed, env seed, track seed or split, max steps. Id shape `PPO_MLP3_REWARD2_SEED42`.
2. Depth-at-fixed-capacity configs: 2, 4, and 6 layer trunks whose approximate parameter counts sit near each other. A unit test checks they are within 20 percent.
3. Capacity ladder configs near 10k, 50k, 100k, 500k, and 1M trunk parameters. Record the count in the file. Do not train them.
4. Reward comparison entries pointing at variants A–D. Algorithm entries for PPO, SAC, and TD3 at the same hidden size. An MLP vs LSTM entry (LSTM via sb3-contrib when that config is selected).
5. `training/evaluate.py` loads a checkpoint, rolls out held-out test seeds, and can aggregate mean and sample spread across run seeds. Default CLI uses a handful of episodes on one or few tracks so it stays short.
6. Write §48 questions under `experiments/`. Mark reward shaping, fixed-capacity depth, and unseen-track generalization as the first three. The other questions stay documented and unanswered.

## Out of scope

Fitting the ladder, the algorithm bake-off, or multi-seed averages. The user has to ask before those runs start.
