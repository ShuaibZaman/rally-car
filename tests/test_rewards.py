"""Reward breakdowns sum, and variant A still pays progress off the track."""

from __future__ import annotations

import unittest

import numpy as np

from environment.env import RallyEnv
from environment.rewards import VARIANTS, compute_reward, get_weights
from environment.types import EnvConfig


class RewardTests(unittest.TestCase):
    def test_breakdown_sums(self) -> None:
        weights = get_weights("D")
        breakdown = compute_reward(
            weights,
            ds=0.4,
            dt=1.0 / 60.0,
            off_track=True,
            collision=False,
            steering=0.5,
            brake=0.2,
            lateral=1.0,
            half_width=8.0,
            curvature=0.03,
            speed=12.0,
            lap_completed=False,
        )
        parts = breakdown.as_dict()
        total = sum(parts[key] for key in parts if key != "total")
        self.assertAlmostEqual(parts["total"], total, places=5)
        self.assertAlmostEqual(breakdown.total, total, places=5)

    def test_collision_dominates_variant_c(self) -> None:
        weights = get_weights("C")
        breakdown = compute_reward(
            weights,
            ds=0.2,
            dt=1.0 / 60.0,
            off_track=True,
            collision=True,
            steering=0.0,
            brake=0.0,
            lateral=9.0,
            half_width=8.0,
            curvature=0.0,
            speed=10.0,
            lap_completed=False,
        )
        self.assertLess(breakdown.collision, -1.0)
        self.assertLess(breakdown.total, breakdown.progress)

    def test_variant_a_pays_progress_off_track(self) -> None:
        env = RallyEnv(EnvConfig(reward="A", preset="easy", track_seed=1_000_000, max_steps=50))
        env.reset(seed=1)
        s = 80.0
        lateral = env.track.half_width + 3.0
        x, y, theta = env.track.pose_at(s, lateral)
        env.car.reset(x, y, theta, speed=12.0)
        env.hint = None
        proj = env.track.project(x, y)
        env.projection = proj
        env.prev_s = proj.s
        env._off_track = True
        _obs, reward, terminated, _truncated, info = env.step(np.array([0.0, 0.2, 0.0], dtype=np.float32))
        self.assertFalse(terminated)
        self.assertFalse(VARIANTS["A"].terminate_collision)
        self.assertGreater(info["breakdown"]["progress"], 0.0)
        self.assertGreater(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
