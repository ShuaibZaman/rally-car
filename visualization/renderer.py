"""Top-down drawing for the lab window. No learning code lives here."""

from __future__ import annotations

import math

import pygame

from environment.track import Track
from environment.types import Ray, VehicleState


class WorldView:
    def __init__(self, rect: pygame.Rect, track: Track) -> None:
        points = track.left
        if len(track.right):
            import numpy as np

            points = np.vstack([track.left, track.right])
        min_x = float(points[:, 0].min())
        max_x = float(points[:, 0].max())
        min_y = float(points[:, 1].min())
        max_y = float(points[:, 1].max())
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        pad = 16
        scale = min((rect.width - pad * 2) / span_x, (rect.height - pad * 2) / span_y)
        self.rect = rect
        self.scale = scale
        self.min_x = min_x
        self.min_y = min_y
        self.pad = pad

    def world(self, x: float, y: float) -> tuple[int, int]:
        sx = self.rect.left + self.pad + (x - self.min_x) * self.scale
        sy = self.rect.bottom - self.pad - (y - self.min_y) * self.scale
        return int(sx), int(sy)


def draw_track(surface: pygame.Surface, track: Track, view: WorldView) -> None:
    grass = pygame.Rect(view.rect)
    pygame.draw.rect(surface, (22, 48, 34), grass)
    left = [view.world(float(p[0]), float(p[1])) for p in track.left]
    right = [view.world(float(p[0]), float(p[1])) for p in track.right]
    if len(left) >= 3:
        pygame.draw.polygon(surface, (58, 62, 72), left)
    pygame.draw.lines(surface, (186, 186, 176), True, left, 2)
    pygame.draw.lines(surface, (186, 186, 176), True, right, 2)
    center = [view.world(float(p[0]), float(p[1])) for p in track.center[:: max(1, len(track.center) // 180)]]
    if len(center) >= 2:
        pygame.draw.lines(surface, (90, 96, 110), True, center, 1)


def draw_polyline(surface: pygame.Surface, points: list[tuple[float, float]], view: WorldView, color: tuple[int, int, int], width: int = 2) -> None:
    if len(points) < 2:
        return
    screen = [view.world(x, y) for x, y in points]
    pygame.draw.lines(surface, color, False, screen, width)


def draw_rays(surface: pygame.Surface, rays: list[Ray], view: WorldView) -> None:
    for ray in rays:
        shade = int(80 + 160 * ray.normalized)
        color = (shade, 180, 220)
        thickness = 1 if ray.normalized > 0.45 else 3
        pygame.draw.line(surface, color, view.world(ray.x0, ray.y0), view.world(ray.x1, ray.y1), thickness)


def draw_car(surface: pygame.Surface, state: VehicleState, view: WorldView, color: tuple[int, int, int] = (232, 196, 72)) -> None:
    length = max(10.0, 3.8 * view.scale)
    width = max(5.0, 1.8 * view.scale)
    cx, cy = view.world(state.x, state.y)
    corners = [(-length * 0.5, -width * 0.5), (length * 0.45, -width * 0.5), (length * 0.45, width * 0.5), (-length * 0.5, width * 0.5)]
    cos_t = math.cos(state.theta)
    sin_t = math.sin(state.theta)
    screen = []
    for lx, ly in corners:
        # Screen y grows downward, and world() already flipped y, so rotate in screen space
        # using the same theta after the flip: x' = x cos - y sin, y' = x sin + y cos, then flip y.
        rx = lx * cos_t - ly * sin_t
        ry = lx * sin_t + ly * cos_t
        screen.append((cx + rx, cy - ry))
    pygame.draw.polygon(surface, color, screen)
    nose = (cx + cos_t * length * 0.45, cy - sin_t * length * 0.45)
    pygame.draw.circle(surface, (30, 30, 30), (int(nose[0]), int(nose[1])), 2)


def draw_heatmap(surface: pygame.Surface, track: Track, hist: list[float], view: WorldView) -> None:
    if not hist or max(hist) <= 0:
        return
    peak = max(hist)
    for index, value in enumerate(hist):
        if value <= 0:
            continue
        s = (index + 0.5) / len(hist) * track.length
        x, y, _heading = track.pose_at(s, 0.0)
        heat = value / peak
        color = (int(80 + 175 * heat), int(180 * (1.0 - heat)), 48)
        pygame.draw.circle(surface, color, view.world(x, y), 5)


def draw_sparkline(surface: pygame.Surface, rect: pygame.Rect, values: list[float], color: tuple[int, int, int], label: str, font: pygame.font.Font) -> None:
    pygame.draw.rect(surface, (16, 20, 28), rect, border_radius=4)
    surface.blit(font.render(label, True, (180, 186, 198)), (rect.x + 8, rect.y + 4))
    if len(values) < 2:
        surface.blit(font.render("no samples yet", True, (120, 126, 140)), (rect.x + 8, rect.y + 28))
        return
    low = min(values)
    high = max(values)
    span = max(high - low, 1e-6)
    points = []
    for i, value in enumerate(values):
        x = rect.x + 8 + (rect.width - 16) * i / (len(values) - 1)
        y = rect.bottom - 8 - (rect.height - 28) * (value - low) / span
        points.append((int(x), int(y)))
    pygame.draw.lines(surface, color, False, points, 2)


def draw_bar(surface: pygame.Surface, origin: tuple[int, int], width: int, fraction: float, color: tuple[int, int, int]) -> None:
    fraction = max(0.0, min(1.0, fraction))
    pygame.draw.rect(surface, (40, 44, 56), pygame.Rect(origin[0], origin[1], width, 14))
    pygame.draw.rect(surface, color, pygame.Rect(origin[0], origin[1], int(width * fraction), 14))


def draw_steering(surface: pygame.Surface, origin: tuple[int, int], width: int, steering: float) -> None:
    pygame.draw.line(surface, (70, 76, 90), (origin[0], origin[1] + 6), (origin[0] + width, origin[1] + 6), 4)
    x = origin[0] + int((steering + 1.0) * 0.5 * width)
    pygame.draw.circle(surface, (232, 196, 72), (x, origin[1] + 6), 7)
