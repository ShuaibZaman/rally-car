"""Sensors, reset determinism, and the classical driver on an easy lap."""

from __future__ import annotations

import unittest

import numpy as np

from agents.classical import PurePursuit, rollout
from environment.env import RallyEnv
from environment.sensors import OBS_DIM, OBS_NAMES, ray_distances
from environment.types import EnvConfig
from environment.vision import render_vision


class SensorTests(unittest.TestCase):
    def test_ray_hits_a_known_segment(self) -> None:
        origin = np.array([0.0, 0.0])
        directions = np.array([[1.0, 0.0], [0.0, 1.0]])
        seg_a = np.array([[5.0, -1.0], [2.0, 4.0]])
        seg_b = np.array([[5.0, 1.0], [2.0, 6.0]])
        distances = ray_distances(origin, directions, seg_a, seg_b, 40.0)
        self.assertAlmostEqual(float(distances[0]), 5.0, places=4)

    def test_observation_width(self) -> None:
        self.assertEqual(len(OBS_NAMES), OBS_DIM)
        self.assertGreaterEqual(OBS_DIM, 20)
        self.assertLessEqual(OBS_DIM, 40)
        env = RallyEnv(EnvConfig(track_seed=1_000_000, preset="easy"))
        obs, _info = env.reset(seed=0)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertGreater(len(env.rays), 0)

    def test_reset_is_deterministic(self) -> None:
        config = EnvConfig(track_seed=1_000_000, preset="easy")
        first = RallyEnv(config)
        second = RallyEnv(config)
        obs_a, _ = first.reset(seed=0)
        obs_b, _ = second.reset(seed=0)
        self.assertTrue(np.allclose(obs_a, obs_b))
        action = np.array([0.2, 0.4, 0.0], dtype=np.float32)
        next_a, reward_a, *_ = first.step(action)
        next_b, reward_b, *_ = second.step(action)
        self.assertTrue(np.allclose(next_a, next_b))
        self.assertAlmostEqual(reward_a, reward_b)

    def test_pure_pursuit_completes_easy_lap(self) -> None:
        env = RallyEnv(EnvConfig(track_seed=1_000_000, preset="easy", reward="C", max_steps=7000))
        expert = PurePursuit()
        metrics, trace, _info = rollout(env, lambda _obs, live: expert.act(live), max_steps=7000)
        self.assertGreater(len(trace), 10)
        self.assertTrue(metrics.completed, msg=f"progress={metrics.progress:.2f} collisions={metrics.collisions}")
        self.assertIsNotNone(metrics.lap_time)
        self.assertLess(metrics.collisions, 1)

    def test_vision_frame_shape(self) -> None:
        env = RallyEnv(EnvConfig(track_seed=1_000_000, preset="easy", observation="vision"))
        obs, _info = env.reset(seed=0)
        self.assertEqual(obs.shape, (84, 84, 3))
        self.assertEqual(obs.dtype, np.uint8)
        frame = render_vision(env.track, env.car.state)
        self.assertEqual(frame.shape, (84, 84, 3))


if __name__ == "__main__":
    unittest.main()
