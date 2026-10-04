"""Shared contracts for the rally lab.

State is (x, y, theta, vx, vy, omega). Actions keep throttle and brake separate.
A track is identified by an integer seed plus a difficulty preset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


def clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0

    @property
    def speed(self) -> float:
        return float((self.vx * self.vx + self.vy * self.vy) ** 0.5)

    def copy(self) -> VehicleState:
        return VehicleState(self.x, self.y, self.theta, self.vx, self.vy, self.omega)


@dataclass
class Action:
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    def clipped(self) -> Action:
        return Action(
            clip(self.steering, -1.0, 1.0),
            clip(self.throttle, 0.0, 1.0),
            clip(self.brake, 0.0, 1.0),
        )

    def as_array(self):
        import numpy as np

        action = self.clipped()
        return np.array([action.steering, action.throttle, action.brake], dtype=np.float32)

    @staticmethod
    def from_array(values) -> Action:
        return Action(float(values[0]), float(values[1]), float(values[2])).clipped()


@dataclass(frozen=True)
class TrackMeta:
    track_id: int
    seed: int
    difficulty: float
    difficulty_name: str
    turns: int
    length_m: float
    width: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PhysicsConfig:
    wheelbase: float = 2.6
    max_steer: float = 0.45
    drive_accel: float = 8.0
    brake_accel: float = 14.0
    drag_linear: float = 0.08
    drag_quadratic: float = 0.008
    grip: float = 1.0
    gravity: float = 9.81
    yaw_tau: float = 0.18
    max_slip: float = 0.22
    accel_scale: float = 1.0


@dataclass
class Projection:
    s: float
    lateral: float
    heading: float
    curvature: float
    index: int


@dataclass
class Ray:
    x0: float
    y0: float
    x1: float
    y1: float
    distance: float
    normalized: float


@dataclass
class RewardBreakdown:
    progress: float = 0.0
    speed: float = 0.0
    centering: float = 0.0
    off_track: float = 0.0
    steering: float = 0.0
    braking: float = 0.0
    collision: float = 0.0
    time: float = 0.0
    lap_bonus: float = 0.0
    racing_line: float = 0.0

    @property
    def total(self) -> float:
        return float(
            self.progress
            + self.speed
            + self.centering
            + self.off_track
            + self.steering
            + self.braking
            + self.collision
            + self.time
            + self.lap_bonus
            + self.racing_line
        )

    def as_dict(self) -> dict:
        data = asdict(self)
        data["total"] = self.total
        return data


@dataclass
class RewardWeights:
    progress: float = 1.0
    off_track: float = 0.0
    collision: float = 0.0
    steering: float = 0.0
    braking: float = 0.0
    time: float = 0.0
    lap_bonus: float = 0.0
    centering: float = 0.0
    racing_line: float = 0.0
    speed: float = 0.0
    terminate_off_track: bool = False
    terminate_collision: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeMetrics:
    completed: bool
    lap_time: float | None
    progress: float
    collisions: int
    avg_speed: float
    off_track_fraction: float
    centerline_deviation: float
    steering_smoothness: float
    throttle_efficiency: float
    recoveries: int
    distance: float
    fuel: float
    steps: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnvConfig:
    preset: str = "easy"
    track_seed: int = 1_000_000
    reward: str = "C"
    dt: float = 1.0 / 60.0
    max_steps: int = 8_000
    observation: str = "sensors"
    surface: str = "asphalt"
    weather: str = "dry"
    obstacles: bool = False
    moving_obstacles: bool = False
    obstacle_count: int = 3
    randomize_start: bool = False
    lateral_force: float = 0.0
    car_half_width: float = 0.9

    def as_dict(self) -> dict:
        return asdict(self)
