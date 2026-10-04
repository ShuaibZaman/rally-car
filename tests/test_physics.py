"""Physics tests: momentum, yaw lag, and braking."""

from __future__ import annotations

import unittest

from environment.physics import integrate
from environment.types import Action, PhysicsConfig, VehicleState


DT = 1.0 / 60.0


def _roll(state: VehicleState, action: Action, seconds: float, config: PhysicsConfig | None = None) -> VehicleState:
    steps = int(seconds / DT)
    for _ in range(steps):
        state = integrate(state, action, DT, config)
    return state


class PhysicsTests(unittest.TestCase):
    def test_coasting_keeps_momentum(self) -> None:
        start = VehicleState(vx=20.0, vy=0.0)
        end = _roll(start, Action(0.0, 0.0, 0.0), 0.2)
        self.assertGreater(end.speed, 15.0)

    def test_steering_does_not_snap_heading(self) -> None:
        config = PhysicsConfig()
        start = VehicleState(vx=20.0, vy=0.0, theta=0.0)
        one = integrate(start, Action(1.0, 0.0, 0.0), DT, config)
        self.assertLess(abs(one.theta), 0.05)
        end = _roll(start, Action(1.0, 0.0, 0.0), 0.1, config)
        self.assertGreater(abs(end.theta), 0.0)
        self.assertLess(abs(end.theta), config.max_steer)

    def test_brake_slows_faster_than_coast(self) -> None:
        start = VehicleState(vx=20.0, vy=0.0)
        coast = _roll(start, Action(0.0, 0.0, 0.0), 0.5)
        braked = _roll(start, Action(0.0, 0.0, 1.0), 0.5)
        self.assertLess(braked.speed, coast.speed)
        self.assertGreaterEqual(braked.speed, 0.0)

    def test_action_clip(self) -> None:
        action = Action(3.0, -1.0, 4.0).clipped()
        self.assertEqual((action.steering, action.throttle, action.brake), (1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
