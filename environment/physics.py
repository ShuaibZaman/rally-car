"""Bicycle integrator with yaw lag and grip-limited lateral acceleration.

Steering sets a yaw-rate command. Heading catches up over yaw_tau, so one
control step cannot snap the car onto the steer angle. Brake removes speed
and does not engage reverse.
"""

from __future__ import annotations

import math

import numpy as np

from environment.types import Action, PhysicsConfig, VehicleState


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def integrate(
    state: VehicleState,
    action: Action,
    dt: float,
    config: PhysicsConfig | None = None,
    grip_scale: float = 1.0,
    lateral_accel: float = 0.0,
) -> VehicleState:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    config = config or PhysicsConfig()
    action = action.clipped()

    speed = state.speed
    steer_angle = action.steering * config.max_steer
    curvature_cmd = math.tan(steer_angle) / config.wheelbase
    omega_cmd = speed * curvature_cmd
    max_lateral = max(config.grip * grip_scale, 0.05) * config.gravity
    max_omega = max_lateral / max(speed, 0.75)
    omega_cmd = float(np.clip(omega_cmd, -max_omega, max_omega))
    blend = 1.0 - math.exp(-dt / config.yaw_tau)
    omega = state.omega + (omega_cmd - state.omega) * blend

    drive = action.throttle * config.drive_accel * config.accel_scale
    brake = action.brake * config.brake_accel
    drag = config.drag_linear * speed + config.drag_quadratic * speed * speed
    new_speed = max(0.0, speed + (drive - brake - drag) * dt)

    theta = wrap_angle(state.theta + omega * dt)
    slip_scale = min(1.0, new_speed / 8.0)
    slip = float(np.clip(-omega * 0.35 * slip_scale, -config.max_slip, config.max_slip))
    vx = new_speed * math.cos(theta + slip)
    vy = new_speed * math.sin(theta + slip)
    # A small world-frame push used by the disturbance scaffold. It does not
    # add a second drivetrain; the next step's speed is recomputed from vx, vy.
    vx += lateral_accel * dt
    return VehicleState(
        x=state.x + vx * dt,
        y=state.y + vy * dt,
        theta=theta,
        vx=float(vx),
        vy=float(vy),
        omega=float(omega),
    )
