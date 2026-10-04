"""Car wrapper around the bicycle integrator."""

from __future__ import annotations

import math
from dataclasses import replace

from environment.physics import integrate
from environment.types import Action, PhysicsConfig, VehicleState


class Car:
    def __init__(self, physics: PhysicsConfig | None = None) -> None:
        self.physics = physics or PhysicsConfig()
        self.state = VehicleState()
        self.prev_action = Action()

    def reset(self, x: float, y: float, theta: float, speed: float = 0.0) -> VehicleState:
        self.state = VehicleState(
            x=float(x),
            y=float(y),
            theta=float(theta),
            vx=float(speed * math.cos(theta)),
            vy=float(speed * math.sin(theta)),
            omega=0.0,
        )
        self.prev_action = Action()
        return self.state

    def step(
        self,
        action: Action,
        dt: float,
        grip_scale: float = 1.0,
        lateral_accel: float = 0.0,
    ) -> VehicleState:
        self.state = integrate(
            self.state,
            action,
            dt,
            self.physics,
            grip_scale=grip_scale,
            lateral_accel=lateral_accel,
        )
        self.prev_action = action.clipped()
        return self.state

    def with_surface(self, grip: float, accel_scale: float, drag_scale: float = 1.0) -> None:
        self.physics = replace(
            self.physics,
            grip=grip,
            accel_scale=accel_scale,
            drag_linear=self.physics.drag_linear * drag_scale,
            drag_quadratic=self.physics.drag_quadratic * drag_scale,
        )
