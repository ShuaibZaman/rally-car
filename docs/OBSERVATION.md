# Observation vector

Sensor driving uses a 24-dimensional `float32` vector. That sits in the 20–40 feature range from spec §5. Vision mode replaces this vector with an 84×84×3 image (spec §6) and is not the default.

Ray values are `distance / 40 m`. `1` means the ray reached max range without a boundary. `0` means the boundary is touching the ray origin. The 15 rays span -100° to +100° relative to the car heading.

| index | name | scale |
| --- | --- | --- |
| 0–14 | `ray_00` … `ray_14` | meters / 40 |
| 15 | `speed` | m/s / 40, clipped to 1.5 |
| 16 | `omega` | rad/s / 2, clipped to ±2 |
| 17 | `heading_error` | radians / π, track heading minus car heading |
| 18 | `center_distance` | signed meters / half width, clipped to ±2 |
| 19 | `curvature` | 1/m / 0.08, clipped to ±3 |
| 20 | `progress` | forward meters / track length, in [0, 1] |
| 21 | `prev_steering` | previous action, [-1, 1] |
| 22 | `prev_throttle` | previous action, [0, 1] |
| 23 | `prev_brake` | previous action, [0, 1] |

Names live in `environment/sensors.py` as `OBS_NAMES`.
