# Architecture experiments

Spec §13, §14, §15.

Two comparisons, both untrained until you ask:

- **Depth, fixed trunk size.** `depth2` `[78, 78]`, `depth4` four layers of 48, `depth6` six layers of 38. Counts are for one MLP from the 24-D observation to 3 actions. Stable-Baselines3 also builds a critic; saved bundles report that full parameter count separately.
- **Capacity ladder.** About 10k, 50k, 100k, 500k, and 1M trunk parameters (`cap_10k` through `cap_1m`).

Also registered, not trained: residual MLP module, CNN, CNN+LSTM (`agents/policies/`).
