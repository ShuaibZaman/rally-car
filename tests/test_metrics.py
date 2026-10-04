"""Score components, experiment ids, and architecture parameter budgets."""

from __future__ import annotations

import unittest

from agents.policies.mlp import trunk_parameter_count
from environment.sensors import OBS_DIM
from environment.types import EpisodeMetrics
from training.catalog import DEPTH_HIDDEN, all_configs
from training.experiment import make_experiment_id
from training.metrics import aggregate_metrics, crash_histogram, describe_trace, rally_score


class MetricTests(unittest.TestCase):
    def test_score_keeps_components(self) -> None:
        metrics = EpisodeMetrics(
            completed=True,
            lap_time=18.0,
            progress=1.0,
            collisions=0,
            avg_speed=20.0,
            off_track_fraction=0.0,
            centerline_deviation=0.4,
            steering_smoothness=0.05,
            throttle_efficiency=10.0,
            recoveries=1,
            distance=400.0,
            fuel=12.0,
            steps=1000,
        )
        scored = rally_score(metrics)
        for key in ("rally_score", "completion_term", "speed_term", "smoothness_term", "crash_term", "off_track_term"):
            self.assertIn(key, scored)
        self.assertGreater(scored["rally_score"], 0.0)

    def test_aggregate_spread(self) -> None:
        rows = [
            rally_score(
                EpisodeMetrics(False, None, 0.2, 1, 5.0, 0.1, 1.0, 0.2, 1.0, 0, 10.0, 1.0, 10)
            ),
            rally_score(
                EpisodeMetrics(True, 20.0, 1.0, 0, 15.0, 0.0, 0.2, 0.05, 8.0, 0, 400.0, 10.0, 800)
            ),
        ]
        summary = aggregate_metrics(rows)
        self.assertEqual(summary["n"], 2)
        self.assertGreater(summary["progress"]["std"], 0.0)
        self.assertEqual(summary["completed_rate"], 0.5)

    def test_behavior_notes_follow_the_trace(self) -> None:
        samples = [
            {"curvature_ahead": 0.05, "brake": 0.0, "speed": 18.0, "steer": 0.0, "collision": False}
            for _ in range(12)
        ]
        notes = describe_trace(samples)
        self.assertIn("Does not brake before corners.", notes)

    def test_corner_markers_follow_the_trace(self) -> None:
        from training.metrics import corner_markers

        samples = [
            {
                "x": float(index),
                "y": 0.0,
                "s": float(index * 5),
                "curvature_ahead": 0.05,
                "brake": 0.0,
                "speed": 18.0,
                "steer": 0.0,
                "collision": False,
            }
            for index in range(12)
        ]
        markers = corner_markers(samples)
        self.assertTrue(markers)
        self.assertEqual(markers[0]["text"], "no brake")

    def test_histogram_bins(self) -> None:
        hist = crash_histogram([{"s": 10.0, "collision": True, "off_track": False}], length=100.0, bins=10)
        self.assertEqual(sum(hist), 1.0)
        self.assertEqual(hist[1], 1.0)


class ExperimentTests(unittest.TestCase):
    def test_experiment_id(self) -> None:
        self.assertEqual(make_experiment_id("PPO", "MLP3", "B", 42), "PPO_MLP3_REWARD2_SEED42")

    def test_depth_budgets_are_close(self) -> None:
        counts = [trunk_parameter_count(OBS_DIM, hidden, 3) for hidden in DEPTH_HIDDEN.values()]
        self.assertLess(max(counts) / min(counts), 1.2)

    def test_catalog_covers_the_matrix(self) -> None:
        names = {config["name"] for config in all_configs()}
        for required in ("depth2", "depth4", "depth6", "cap_10k", "cap_1m", "reward_A", "algo_ppo", "algo_sac", "algo_td3", "memory_lstm", "vision_cnn", "generalization_holdout"):
            self.assertIn(required, names)
        for config in all_configs():
            self.assertFalse(config["train"])


if __name__ == "__main__":
    unittest.main()
