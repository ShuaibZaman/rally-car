"""Track generation is a pure function of seed and preset."""

from __future__ import annotations

import unittest

import numpy as np

from environment.generator import (
    SPLIT_START,
    generate_track,
    split_seeds,
)


class TrackTests(unittest.TestCase):
    def test_same_seed_matches(self) -> None:
        first = generate_track(918273, "medium")
        second = generate_track(918273, "medium")
        self.assertTrue(np.allclose(first.center, second.center))
        self.assertEqual(first.width, second.width)
        self.assertTrue(np.allclose(first.checkpoint_s, second.checkpoint_s))
        self.assertEqual(first.meta.seed, 918273)
        self.assertEqual(first.meta.difficulty_name, "medium")
        self.assertGreater(first.meta.length_m, 100.0)

    def test_presets_differ(self) -> None:
        easy = generate_track(918273, "easy")
        hard = generate_track(918273, "hard")
        same_shape = easy.center.shape == hard.center.shape
        self.assertFalse(same_shape and np.allclose(easy.center, hard.center))
        self.assertGreater(easy.width, hard.width)

    def test_pools_are_disjoint_and_sized(self) -> None:
        train = set(split_seeds("train"))
        val = set(split_seeds("val"))
        test = set(split_seeds("test"))
        self.assertEqual(len(train), 1000)
        self.assertEqual(len(val), 100)
        self.assertEqual(len(test), 100)
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(min(train), SPLIT_START["train"])

    def test_projection_on_centerline_is_near_zero(self) -> None:
        track = generate_track(1_000_000, "easy")
        x, y, _heading = track.pose_at(40.0, 0.0)
        proj = track.project(x, y)
        self.assertLess(abs(proj.lateral), 0.05)
        self.assertLess(abs(proj.s - 40.0), 1.0)


if __name__ == "__main__":
    unittest.main()
