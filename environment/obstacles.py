"""Static and moving obstacles. The env ignores them unless the flag is on."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment.track import Track


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0

    def step(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt


def sample_obstacles(track: Track, seed: int, count: int = 3, moving: bool = False) -> list[Obstacle]:
    rng = np.random.default_rng(int(seed) + 901)
    obstacles: list[Obstacle] = []
    for _ in range(count):
        s = float(rng.uniform(track.length * 0.15, track.length * 0.9))
        lateral = float(rng.uniform(-track.half_width * 0.45, track.half_width * 0.45))
        x, y, heading = track.pose_at(s, lateral)
        vx = vy = 0.0
        if moving:
            pace = float(rng.uniform(2.0, 6.0))
            vx = pace * float(np.cos(heading))
            vy = pace * float(np.sin(heading))
        obstacles.append(Obstacle(x=x, y=y, radius=1.6, vx=vx, vy=vy))
    return obstacles
