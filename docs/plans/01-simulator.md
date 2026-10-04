# 01 — Simulator

Spec: §3, §4, §50 phase 1. Contracts in `environment/types.py`.

The car must have momentum. One control step must not snap heading to the steering command. Manual keyboard driving uses the same action triple as every later agent.

## Steps

1. Define `VehicleState` (`x`, `y`, `theta`, `vx`, `vy`, `omega`) and `Action` (`steering`, `throttle`, `brake`) with clipping.
2. Implement a bicycle integrator in `environment/physics.py`: longitudinal accel from throttle, brake, and drag; yaw command `v * tan(steer) / wheelbase`; yaw first-order lag; lateral accel limited by grip. Brake does not reverse the car.
3. Wrap the integrator in `environment/car.py` with the previous action stored for the observation.
4. Detect off-track vs collision from lateral offset against the track half-width (center outside the ribbon is a collision).
5. Place ordered checkpoints on arc length. Progress is forward distance along the centerline. A lap completes only after the car has advanced about one full length without skipping the checkpoint sequence.
6. Top-down camera that fits the whole track in the live panel, plus a keyboard loop: steer, throttle, brake, reset.
7. Tests: coasting keeps most of the speed over a short interval; full steering changes heading smoothly and does not equal the steer angle after a tenth of a second; brake slows the car faster than coasting.

## Out of scope

Raycasts, rewards, and learning. Those are plans 03–05.
