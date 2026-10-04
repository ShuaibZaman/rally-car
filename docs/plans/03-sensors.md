# 03 — Sensors and classical baseline

Spec: §5, §21, §30, §50 phase 2.

The observation is numerical and documented. The classical driver is the non-neural baseline and the behavior-cloning expert.

## Steps

1. Cast 15 rays from the car, spread across the forward hemisphere, against both boundaries. Normalize distance by max range so 1 is clear and 0 is contact. Keep the world-space segments for drawing.
2. Append kinematic features: speed, angular velocity, heading error, signed center distance, curvature, progress, previous steering, previous throttle, previous brake. Document the full vector in `docs/OBSERVATION.md`. Dimension must stay inside 20–40.
3. Draw the rays in the live view. Thickness or color reflects distance.
4. Gymnasium env: `Box` action `[steer, throttle, brake]`, the sensor observation, reset from a track seed, deterministic `reset(seed)`.
5. Pure pursuit: lookahead grows with speed, steering tracks the lookahead point, throttle drops and brake rises with upcoming curvature and with a close forward ray.
6. Test: observation length and finiteness; two resets with the same seed match; on the easy fixture the pure-pursuit driver completes a lap.

## Out of scope

Pixel observations. Those stay a scaffold in plan 08 until a sensor policy already drives.
