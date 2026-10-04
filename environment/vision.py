"""84x84 top-down image for the vision scaffold. Not used by the default trainer."""

from __future__ import annotations

import math

import numpy as np

from environment.obstacles import Obstacle
from environment.track import Track
from environment.types import VehicleState


def _draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    height, width, _ = image.shape
    steps = int(max(abs(x1 - x0), abs(y1 - y0), 1))
    for t in range(steps + 1):
        x = int(x0 + (x1 - x0) * t / steps)
        y = int(y0 + (y1 - y0) * t / steps)
        if 0 <= x < width and 0 <= y < height:
            image[y, x] = color


def render_vision(
    track: Track,
    state: VehicleState,
    obstacles: list[Obstacle] | None = None,
    size: int = 84,
    meters_per_pixel: float = 0.8,
) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = (34, 92, 48)
    half = size * 0.5

    def to_px(x: float, y: float) -> tuple[int, int]:
        col = int(half + (x - state.x) / meters_per_pixel)
        row = int(half - (y - state.y) / meters_per_pixel)
        return col, row

    def polyline(points: np.ndarray, color: tuple[int, int, int]) -> None:
        for i in range(len(points)):
            x0, y0 = to_px(float(points[i, 0]), float(points[i, 1]))
            j = (i + 1) % len(points)
            x1, y1 = to_px(float(points[j, 0]), float(points[j, 1]))
            if (0 <= x0 < size and 0 <= y0 < size) or (0 <= x1 < size and 0 <= y1 < size):
                _draw_line(image, x0, y0, x1, y1, color)

    polyline(track.left, (210, 210, 210))
    polyline(track.right, (210, 210, 210))
    for obstacle in obstacles or []:
        col, row = to_px(obstacle.x, obstacle.y)
        r = max(1, int(obstacle.radius / meters_per_pixel))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    x = col + dx
                    y = row + dy
                    if 0 <= x < size and 0 <= y < size:
                        image[y, x] = (90, 90, 90)
    nose = (
        state.x + math.cos(state.theta) * 2.2,
        state.y + math.sin(state.theta) * 2.2,
    )
    tail = (
        state.x - math.cos(state.theta) * 1.6,
        state.y - math.sin(state.theta) * 1.6,
    )
    _draw_line(image, *to_px(*tail), *to_px(*nose), (240, 200, 60))
    return image
