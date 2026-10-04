"""Scaffolds: surfaces, curriculum, obstacles, PPO loss, behavior cloning, networks."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from agents.behavior_cloning import (
    LOSS_NAMES,
    collect_demonstrations,
    copy_matching_parameters,
    regression_loss,
    train_bc_step,
)
from agents.policies.cnn import DrivingCNN
from agents.policies.cnn_lstm import CnnLstm
from agents.policies.mlp import MLP
from agents.policies.residual_mlp import ResidualMLP
from agents.ppo import ppo_loss
from environment.curriculum import STAGES, stage_config
from environment.env import RallyEnv
from environment.generator import generate_track
from environment.obstacles import sample_obstacles
from environment.surfaces import SURFACES, combined_surface
from environment.types import EnvConfig
from environment.weather import weather_names
from visualization.dashboard import action_from_flags


class ExtensionTests(unittest.TestCase):
    def test_surfaces_and_weather(self) -> None:
        self.assertGreater(SURFACES["asphalt"]["grip"], SURFACES["snow"]["grip"])
        wet = combined_surface("asphalt", "rain")
        dry = combined_surface("asphalt", "dry")
        self.assertLess(wet["grip"], dry["grip"])
        self.assertEqual(weather_names(), ["dry", "rain", "snow"])

    def test_curriculum_order(self) -> None:
        self.assertEqual([stage["stage"] for stage in STAGES], list(range(1, 8)))
        last = stage_config(7)
        self.assertTrue(last["randomize_start"])
        self.assertGreater(last["lateral_force"], 0.0)

    def test_obstacles_are_deterministic(self) -> None:
        track = generate_track(1_000_000, "easy")
        first = sample_obstacles(track, 5, count=3, moving=True)
        second = sample_obstacles(track, 5, count=3, moving=True)
        self.assertEqual([(item.x, item.y, item.vx) for item in first], [(item.x, item.y, item.vx) for item in second])

    def test_ppo_loss_is_finite(self) -> None:
        new_logp = torch.zeros(8, requires_grad=True)
        old_logp = torch.zeros(8)
        advantages = torch.linspace(-1, 1, 8)
        values = torch.zeros(8, requires_grad=True)
        returns = torch.ones(8)
        entropy = torch.ones(8)
        losses = ppo_loss(new_logp, old_logp, advantages, values, returns, entropy)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(new_logp.grad)

    def test_behavior_cloning_step(self) -> None:
        env = RallyEnv(EnvConfig(track_seed=1_000_000, preset="easy", max_steps=20))
        observations, actions = collect_demonstrations(env, steps=8)
        self.assertEqual(observations.shape[0], 8)
        self.assertEqual(actions.shape[1], 3)
        model = MLP(observations.shape[1], [16], 3)
        for name in LOSS_NAMES:
            value = train_bc_step(model, observations, actions, "Adam", name)
            self.assertTrue(np.isfinite(value))
        other = MLP(observations.shape[1], [16], 3)
        copied = copy_matching_parameters(model, other)
        self.assertGreater(copied, 0)
        pred = model(torch.zeros(2, observations.shape[1]))
        target = torch.zeros_like(pred)
        self.assertGreaterEqual(float(regression_loss(pred, target, "huber").detach()), 0.0)

    def test_policy_modules_forward(self) -> None:
        batch = torch.zeros(2, 24)
        self.assertEqual(MLP(24, [32, 32], 3)(batch).shape, (2, 3))
        self.assertEqual(ResidualMLP(24, 32, blocks=2, out_dim=3)(batch).shape, (2, 3))
        image = torch.zeros(2, 84, 84, 3)
        self.assertEqual(DrivingCNN()(image).shape, (2, 3))
        sequence = image[:, None].repeat(1, 2, 1, 1, 1)
        actions, _state = CnnLstm()(sequence)
        self.assertEqual(actions.shape, (2, 2, 3))

    def test_keyboard_flags(self) -> None:
        action = action_from_flags(left=False, right=True, throttle=True, brake=False)
        self.assertEqual(action.steering, 1.0)
        self.assertEqual(action.throttle, 1.0)
        self.assertEqual(action.brake, 0.0)


if __name__ == "__main__":
    unittest.main()
