"""Fifteen forward raycasts plus kinematic features.

Ray values are distance / max range: 1 is clear, 0 is contact.
The full vector is documented in docs/OBSERVATION.md.
"""

from __future__ import annotations

import math

import numpy as np

from environment.physics import wrap_angle
from environment.track import Track
from environment.types import Ray, VehicleState

N_RAYS = 15
MAX_RAY_M = 40.0
RAY_ANGLES = np.linspace(math.radians(-100.0), math.radians(100.0), N_RAYS)
SPEED_SCALE = 40.0
OMEGA_SCALE = 2.0
CURVATURE_SCALE = 0.08

OBS_NAMES = [
    *[f"ray_{i:02d}" for i in range(N_RAYS)],
    "speed",
    "omega",
    "heading_error",
    "center_distance",
    "curvature",
    "progress",
    "prev_steering",
    "prev_throttle",
    "prev_brake",
]
OBS_DIM = len(OBS_NAMES)


def ray_distances(
    origin: np.ndarray,
    directions: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    """Vectorized ray vs segment distances. Shape (n_rays,)."""
    v = seg_b - seg_a
    denom = directions[:, 0:1] * v[None, :, 1] - directions[:, 1:2] * v[None, :, 0]
    diff = seg_a[None, :, :] - origin.reshape(1, 1, 2)
    safe = np.where(np.abs(denom) < 1e-8, np.nan, denom)
    t = (diff[:, :, 0] * v[None, :, 1] - diff[:, :, 1] * v[None, :, 0]) / safe
    u = (diff[:, :, 0] * directions[:, 1:2] - diff[:, :, 1] * directions[:, 0:1]) / safe
    valid = (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = np.where(valid, t, np.inf)
    nearest = np.min(t, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, max_dist)
    return np.clip(nearest, 0.0, max_dist).astype(np.float64)


def _closed_segments(polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return polyline, np.roll(polyline, -1, axis=0)


def cast_rays(x: float, y: float, theta: float, track: Track) -> list[Ray]:
    origin = np.array([x, y], dtype=np.float64)
    directions = np.stack(
        [np.cos(theta + RAY_ANGLES), np.sin(theta + RAY_ANGLES)],
        axis=1,
    )
    left_a, left_b = _closed_segments(track.left)
    right_a, right_b = _closed_segments(track.right)
    seg_a = np.vstack([left_a, right_a])
    seg_b = np.vstack([left_b, right_b])
    distances = ray_distances(origin, directions, seg_a, seg_b, MAX_RAY_M)
    rays: list[Ray] = []
    for direction, distance in zip(directions, distances):
        hit = origin + direction * float(distance)
        rays.append(
            Ray(
                x0=float(origin[0]),
                y0=float(origin[1]),
                x1=float(hit[0]),
                y1=float(hit[1]),
                distance=float(distance),
                normalized=float(distance / MAX_RAY_M),
            )
        )
    return rays


def build_observation(
    state: VehicleState,
    prev_steering: float,
    prev_throttle: float,
    prev_brake: float,
    heading: float,
    lateral: float,
    curvature: float,
    half_width: float,
    progress: float,
    rays: list[Ray],
) -> np.ndarray:
    heading_error = wrap_angle(heading - state.theta)
    values = [
        *[ray.normalized for ray in rays],
        float(np.clip(state.speed / SPEED_SCALE, 0.0, 1.5)),
        float(np.clip(state.omega / OMEGA_SCALE, -2.0, 2.0)),
        float(heading_error / math.pi),
        float(np.clip(lateral / max(half_width, 1e-6), -2.0, 2.0)),
        float(np.clip(curvature / CURVATURE_SCALE, -3.0, 3.0)),
        float(np.clip(progress, 0.0, 1.0)),
        float(prev_steering),
        float(prev_throttle),
        float(prev_brake),
    ]
    obs = np.asarray(values, dtype=np.float32)
    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"expected observation {OBS_DIM}, got {obs.shape}")
    return obs


def neutral_observation(speed: float = 0.0, curvature: float = 0.0) -> np.ndarray:
    """Sensor vector with clear rays and two varying features, for policy slices."""
    rays = [Ray(0, 0, 0, 0, MAX_RAY_M, 1.0) for _ in range(N_RAYS)]
    state = VehicleState(vx=speed, vy=0.0)
    return build_observation(
        state,
        0.0,
        0.0,
        0.0,
        heading=0.0,
        lateral=0.0,
        curvature=curvature,
        half_width=8.0,
        progress=0.0,
        rays=rays,
    )
