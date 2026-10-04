# 07 — Presentation

Spec: §19, §20, §22, §24, §42, §43, §44, §45, §46, §47.

Everything on screen comes from a recorded trace, a checkpoint, or an eval JSON. Missing runs render as empty states.

## Steps

1. Save `(x, y, t)` during a rollout and draw the polyline on the track. Overlay more than one trace when several are loaded (§19).
2. Ghost mode: classical controller, uniform random actions, and a loaded policy, same seed and start pose. Show each lap time or "dnf" (§20, §42).
3. Timeline: list checkpoint files, left/right to change which policy is replayed (§23). One smoke checkpoint is a valid one-stop timeline.
4. Evolution view with six slots (random, beginner, competent, advanced, racing line, unseen). A slot without a file says so (§24).
5. Crash heatmap: bin collision and off-track samples by arc length and draw them on the centerline (§46).
6. Policy slice: grid over normalized speed and curvature, other features neutral, plot brake from the loaded network (§45). Matplotlib figure under `data/results/`.
7. Compare view: read eval JSON files and show completion, lap time, collisions, off-track, and parameter count (§35 D, §43). Behavior notes (§44) are sentences computed from the trace (brake before high curvature, exit speed), included only when the trace supports them.
8. Generalization caption (§47): training-split metric vs test-split metric when both JSON files exist.

## Out of scope

A web frontend. That is plan 08 and stays a README until the pygame lab is the one people actually run.
