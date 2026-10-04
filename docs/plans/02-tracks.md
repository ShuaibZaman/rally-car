# 02 — Tracks

Spec: §7, §8, §50 phase 4.

Every track is a closed curve plus a metadata card. Generation is a pure function of `(seed, preset)`.

## Steps

1. Represent a track as arc-length-sampled centerline, left and right boundaries, heading, curvature, width, and checkpoint distances.
2. Easy preset: a stadium (two straights, two arcs) with a small seeded jitter of straight length and radius. Width stays wide.
3. Medium, hard, and extreme: closed radial curves. Parameters that change with difficulty: width, harmonic count, amplitude, base radius. Retry with smaller amplitude if the ribbon self-intersects. The retry rule is deterministic.
4. Metadata card: track id, seed, difficulty in `[0, 1]`, difficulty name, turn count, length in meters, width.
5. Disjoint seed pools: train 1000, validation 100, test 100. Tests use a fixture subset, not the full pools. Test seeds never appear in the train pool.
6. `project(x, y, hint)` returns arc length, signed lateral offset, heading, and curvature. Pose-at-s places the car for reset.
7. Test: two builds from the same seed and preset match centerline, width, and checkpoints. Different presets from one seed differ. Pools are disjoint.

## Out of scope

Rendering style and the learning split's actual training runs. The pools only have to exist and be reproducible.
