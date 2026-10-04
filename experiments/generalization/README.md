# Generalization

Spec §8, §47.

Seed pools do not overlap:

- train: `1000000`–`1000999` (1000 tracks)
- validation: `2000000`–`2000099` (100 tracks)
- test: `3000000`–`3000099` (100 tracks)

The smoke run uses train seed `1000000` only. `training/evaluate.py --split test` scores whatever checkpoint you pass on held-out seeds. A training-split JSON and a test-split JSON can sit side by side; the dashboard compare view reads `data/results/eval_*.json`.

Same seed and preset rebuild the same centerline. That is the reproducibility hook in spec §7 and §38.
