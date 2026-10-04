# 04 — Rewards

Spec: §9, §10, §11, §27, §52.

Reward terms and training losses are different objects. Variant configs do not set the optimizer.

## Steps

1. Compute additive terms: progress, off-track, collision, excessive steering, excessive braking, time, lap bonus, centering, and racing-line (ideal offset toward the inside of a curve).
2. Publish a per-step breakdown whose fields sum to the scalar reward, for the HUD in §22.
3. Variant A — progress only. Do not terminate for leaving the track. This is the reward-hacking fixture (§27).
4. Variant B — progress minus collision. Variant C — progress, collision, time, off-track (default racing objective). Variant D — C plus the racing-line term.
5. Store A–D as data files under `training/configs/rewards/`. They name weights only. Algorithm, architecture, and learning rate live in other configs (§11).
6. Test: breakdown total equals the scalar; variant A still pays progress while off track; variant C's collision term dominates a crash step.

## Out of scope

Training the four variants against each other. Plan 06 only registers the comparison.
