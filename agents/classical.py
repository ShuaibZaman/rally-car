"""Pure-pursuit steering and a curvature-based throttle.

This is the non-neural baseline and the behavior-cloning expert.
"""

from __future__ import annotations

import math

import numpy as np

from environment.physics import wrap_angle


class PurePursuit:
    def __init__(self, max_steer: float = 0.45) -> None:
        self.max_steer = max_steer

    def act(self, env) -> np.ndarray:
        state = env.car.state
        track = env.track
        proj = env.projection
        speed = state.speed
        lookahead = 10.0 + 0.4 * speed
        target_s = (proj.s + lookahead) % track.length
        tx, ty, _ = track.pose_at(target_s, 0.0)
        error = wrap_angle(math.atan2(ty - state.y, tx - state.x) - state.theta)
        steer = float(np.clip(error / self.max_steer, -1.0, 1.0))

        curvature = track.max_curvature_ahead(proj.s, max(lookahead, 20.0))
        grip = env.car.physics.grip * env.car.physics.gravity
        limit = math.sqrt(grip / max(curvature, 1e-3)) * 0.92
        throttle = 0.55
        brake = 0.0
        if speed > limit:
            throttle = 0.0
            brake = float(np.clip((speed - limit) / 6.0, 0.0, 1.0))
        elif speed < limit * 0.72:
            throttle = 0.9
        if abs(error) > 0.35:
            throttle = min(throttle, 0.35)
        if env.rays:
            front = min(ray.normalized for ray in env.rays[5:10])
            if front < 0.18:
                throttle = 0.0
                brake = max(brake, 0.85)
        return np.array([steer, throttle, brake], dtype=np.float32)


def rollout(env, policy, max_steps: int | None = None):
    """Run one episode. `policy(obs, env) -> action`."""
    obs, info = env.reset()
    done = False
    steps = 0
    limit = max_steps or env.config.max_steps
    last = info
    while not done and steps < limit:
        action = policy(obs, env)
        obs, _reward, terminated, truncated, last = env.step(action)
        done = terminated or truncated
        steps += 1
    return env.episode_metrics(), env.trace, last
