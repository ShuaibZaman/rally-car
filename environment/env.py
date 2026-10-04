"""Gymnasium rally environment.

Sensor observations are the default. Vision mode returns an 84x84x3 image and
is refused by the trainer unless --allow-vision is set.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.car import Car
from environment.generator import generate_track
from environment.obstacles import sample_obstacles
from environment.physics import wrap_angle
from environment.rewards import compute_reward, get_weights
from environment.sensors import OBS_DIM, build_observation, cast_rays
from environment.surfaces import combined_surface
from environment.types import Action, EnvConfig, EpisodeMetrics, Projection, Ray


def _crossed(prev_s: float, new_s: float, checkpoint: float, ds: float) -> bool:
    if ds <= 0.0:
        return False
    if new_s >= prev_s:
        return prev_s < checkpoint <= new_s
    return checkpoint > prev_s or checkpoint <= new_s


class RallyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.weights = get_weights(self.config.reward)
        if self.config.observation == "vision":
            self.observation_space = spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)
        elif self.config.observation == "sensors":
            self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        else:
            raise ValueError(self.config.observation)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.car = Car()
        self.obstacles = []
        self.track = generate_track(self.config.track_seed, self.config.preset)
        self.rays: list[Ray] = []
        self.projection = Projection(0.0, 0.0, 0.0, 0.0, 0)
        self.trace: list[dict] = []
        self._apply_surface()
        self._clear_episode()

    def _apply_surface(self) -> None:
        scales = combined_surface(self.config.surface, self.config.weather)
        base_linear = 0.08
        base_quad = 0.008
        from environment.types import PhysicsConfig

        self.car.physics = PhysicsConfig(
            grip=scales["grip"],
            accel_scale=scales["accel"],
            drag_linear=base_linear * scales["drag"],
            drag_quadratic=base_quad * scales["drag"],
        )

    def _clear_episode(self) -> None:
        self.steps = 0
        self.time = 0.0
        self.forward_m = 0.0
        self.prev_s = 0.0
        self.hint: int | None = None
        self.checkpoints_hit = 0
        self.remaining: list[float] = []
        self.lap_completed = False
        self.collision_count = 0
        self.off_steps = 0
        self.speed_sum = 0.0
        self.lateral_sum = 0.0
        self.steer_delta_sum = 0.0
        self.prev_steer = 0.0
        self.distance = 0.0
        self.fuel = 0.0
        self.recoveries = 0
        self.was_off = False
        self.last_reward = 0.0
        self.last_breakdown: dict = {}
        self.trace = []
        self._off_track = False
        self._collision = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._clear_episode()
        self._apply_surface()
        track_seed = self.config.track_seed
        preset = self.config.preset
        if options:
            track_seed = int(options.get("track_seed", track_seed))
            preset = options.get("preset", preset)
        self.track = generate_track(track_seed, preset)
        start_s = 2.0
        lateral = 0.0
        speed = 0.0
        if self.config.randomize_start:
            rng = np.random.default_rng(0 if seed is None else seed)
            start_s = float(rng.uniform(0.0, self.track.length))
            lateral = float(rng.uniform(-1.0, 1.0))
            speed = float(rng.uniform(0.0, 8.0))
        x, y, theta = self.track.pose_at(start_s, lateral)
        self.car.reset(x, y, theta, speed)
        self.projection = self.track.project(x, y)
        self.prev_s = self.projection.s
        self.hint = self.projection.index
        order = [float(s) for s in self.track.checkpoint_s if s > self.prev_s + 1e-3]
        order += [float(s) for s in self.track.checkpoint_s if s <= self.prev_s + 1e-3]
        self.remaining = order
        self.obstacles = []
        if self.config.obstacles:
            self.obstacles = sample_obstacles(
                self.track,
                track_seed,
                self.config.obstacle_count,
                moving=self.config.moving_obstacles,
            )
        self.rays = cast_rays(x, y, theta, self.track)
        obs = self._observe()
        return obs, self._info()

    def step(self, action):
        control = Action.from_array(np.asarray(action, dtype=np.float32))
        grip_scale = 0.45 if self._off_track else 1.0
        self.car.step(
            control,
            self.config.dt,
            grip_scale=grip_scale,
            lateral_accel=self.config.lateral_force,
        )
        if self.config.moving_obstacles:
            for obstacle in self.obstacles:
                obstacle.step(self.config.dt)
        state = self.car.state
        self.projection = self.track.project(state.x, state.y, hint=self.hint)
        self.hint = self.projection.index
        ds = (self.projection.s - self.prev_s) % self.track.length
        if ds > self.track.length * 0.5:
            ds -= self.track.length
        if abs(ds) > 30.0:
            ds = 0.0
        self.forward_m += max(ds, 0.0)
        self._pass_checkpoints(self.prev_s, self.projection.s, ds)
        self.prev_s = self.projection.s

        half = self.track.half_width
        self._off_track = abs(self.projection.lateral) > (half - self.config.car_half_width)
        self._collision = abs(self.projection.lateral) > half
        if self.config.obstacles:
            for obstacle in self.obstacles:
                dist = math.hypot(obstacle.x - state.x, obstacle.y - state.y)
                if dist < obstacle.radius + self.config.car_half_width:
                    self._collision = True
        if self._collision:
            self.collision_count += 1
        if self.was_off and not self._off_track:
            self.recoveries += 1
        self.was_off = self._off_track

        self.lap_completed = self.checkpoints_hit >= len(self.track.checkpoint_s) and self.forward_m >= self.track.length * 0.98
        breakdown = compute_reward(
            self.weights,
            ds=ds,
            dt=self.config.dt,
            off_track=self._off_track,
            collision=self._collision,
            steering=control.steering,
            brake=control.brake,
            lateral=self.projection.lateral,
            half_width=half,
            curvature=self.projection.curvature,
            speed=state.speed,
            lap_completed=self.lap_completed,
        )
        self.last_breakdown = breakdown.as_dict()
        self.last_reward = breakdown.total
        self.steps += 1
        self.time += self.config.dt
        self.speed_sum += state.speed
        self.lateral_sum += abs(self.projection.lateral)
        self.steer_delta_sum += abs(control.steering - self.prev_steer)
        self.prev_steer = control.steering
        self.distance += state.speed * self.config.dt
        self.fuel += control.throttle * self.config.dt
        if self._off_track:
            self.off_steps += 1

        self.rays = cast_rays(state.x, state.y, state.theta, self.track)
        ahead = self.track.max_curvature_ahead(self.projection.s, 18.0)
        self.trace.append(
            {
                "x": state.x,
                "y": state.y,
                "t": self.time,
                "s": self.projection.s,
                "speed": state.speed,
                "steer": control.steering,
                "throttle": control.throttle,
                "brake": control.brake,
                "collision": self._collision,
                "off_track": self._off_track,
                "lateral": self.projection.lateral,
                "curvature": self.projection.curvature,
                "curvature_ahead": ahead,
                "reward": self.last_reward,
                "heading_error": wrap_angle(self.projection.heading - state.theta),
            }
        )

        terminated = False
        if self._collision and self.weights.terminate_collision:
            terminated = True
        if self._off_track and self.weights.terminate_off_track:
            terminated = True
        if self.lap_completed:
            terminated = True
        truncated = self.steps >= self.config.max_steps
        return self._observe(), float(self.last_reward), terminated, truncated, self._info()

    def _pass_checkpoints(self, prev_s: float, new_s: float, ds: float) -> None:
        guard = 0
        while self.remaining and _crossed(prev_s, new_s, self.remaining[0], ds) and guard < 8:
            self.remaining.pop(0)
            self.checkpoints_hit += 1
            guard += 1

    def _observe(self):
        if self.config.observation == "vision":
            from environment.vision import render_vision

            return render_vision(self.track, self.car.state, self.obstacles)
        progress = self.progress_fraction
        return build_observation(
            self.car.state,
            self.car.prev_action.steering,
            self.car.prev_action.throttle,
            self.car.prev_action.brake,
            self.projection.heading,
            self.projection.lateral,
            self.projection.curvature,
            self.track.half_width,
            progress,
            self.rays,
        )

    @property
    def progress_fraction(self) -> float:
        if self.lap_completed:
            return 1.0
        if self.track.length <= 0:
            return 0.0
        return float(min(0.999, self.forward_m / self.track.length))

    def episode_metrics(self) -> EpisodeMetrics:
        n = max(self.steps, 1)
        return EpisodeMetrics(
            completed=bool(self.lap_completed),
            lap_time=float(self.time) if self.lap_completed else None,
            progress=self.progress_fraction,
            collisions=int(self.collision_count),
            avg_speed=float(self.speed_sum / n),
            off_track_fraction=float(self.off_steps / n),
            centerline_deviation=float(self.lateral_sum / n),
            steering_smoothness=float(self.steer_delta_sum / max(n - 1, 1)),
            throttle_efficiency=float(self.distance / (self.fuel + 0.1)),
            recoveries=int(self.recoveries),
            distance=float(self.distance),
            fuel=float(self.fuel),
            steps=int(self.steps),
        )

    def _info(self) -> dict:
        metrics = self.episode_metrics()
        return {
            "progress": metrics.progress,
            "lap_time": metrics.lap_time if metrics.lap_time is not None else -1.0,
            "collision": self._collision,
            "off_track": self._off_track,
            "speed": self.car.state.speed,
            "lap_completed": self.lap_completed,
            "breakdown": self.last_breakdown,
            "track_seed": self.track.meta.seed,
            "metrics": metrics.as_dict(),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            from environment.vision import render_vision

            return render_vision(self.track, self.car.state, self.obstacles)
        return None
