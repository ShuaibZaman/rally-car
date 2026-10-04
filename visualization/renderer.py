"""Top-down drawing for the lab window. No learning code lives here."""

from __future__ import annotations

import math

import pygame

from environment.track import Track
from environment.types import Ray, VehicleState

# Graphite cockpit. Channel colors stay the same in every panel.
BG = (7, 8, 10)
PANEL = (16, 18, 22)
STROKE = (46, 50, 58)
TEXT = (226, 228, 232)
MUTED = (138, 144, 156)
GOLD = (214, 176, 64)
THROTTLE = (64, 196, 122)
BRAKE = (214, 78, 72)
GRASS = (16, 40, 28)
ASPHALT = (52, 54, 60)
EDGE = (214, 210, 196)
CENTER = (86, 90, 98)
START = (232, 208, 96)
CLASSICAL = (86, 186, 224)
RANDOM = (154, 158, 166)
LEARNED = (232, 196, 72)


def mono(size: int) -> pygame.font.Font:
    matched = pygame.font.match_font("consolas") or pygame.font.match_font("cascadiamono")
    if matched:
        return pygame.font.Font(matched, size)
    return pygame.font.SysFont("consolas", size)


def panel(surface: pygame.Surface, rect: pygame.Rect) -> None:
    pygame.draw.rect(surface, PANEL, rect)
    pygame.draw.rect(surface, STROKE, rect, 1)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def speed_color(speed: float, vmax: float = 28.0) -> tuple[int, int, int]:
    t = _clip(speed / vmax, 0.0, 1.0)
    if t < 0.5:
        u = t * 2.0
        return (int(48 + 180 * u), int(110 + 90 * u), int(210 - 150 * u))
    u = (t - 0.5) * 2.0
    return (int(228), int(200 - 150 * u), int(60 - 30 * u))


def ray_color(normalized: float) -> tuple[int, int, int]:
    t = _clip(normalized, 0.0, 1.0)
    return (
        int(220 * (1.0 - t) + 88 * t),
        int(64 * (1.0 - t) + 118 * t),
        int(56 * (1.0 - t) + 128 * t),
    )


class WorldView:
    def __init__(self, rect: pygame.Rect, track: Track, focus: tuple[float, float] | None = None, span_m: float = 78.0) -> None:
        import numpy as np

        self.rect = rect
        self.pad = 14
        if focus is None:
            points = np.vstack([track.left, track.right])
            min_x = float(points[:, 0].min())
            max_x = float(points[:, 0].max())
            min_y = float(points[:, 1].min())
            max_y = float(points[:, 1].max())
            span_x = max(max_x - min_x, 1.0)
            span_y = max(max_y - min_y, 1.0)
            self.scale = min((rect.width - self.pad * 2) / span_x, (rect.height - self.pad * 2) / span_y)
            self.min_x = min_x
            self.min_y = min_y
        else:
            usable_w = max(rect.width - self.pad * 2, 1)
            usable_h = max(rect.height - self.pad * 2, 1)
            self.scale = min(usable_w, usable_h) / span_m
            half_w = usable_w / self.scale / 2.0
            half_h = usable_h / self.scale / 2.0
            self.min_x = focus[0] - half_w
            self.min_y = focus[1] - half_h

    def world(self, x: float, y: float) -> tuple[int, int]:
        sx = self.rect.left + self.pad + (x - self.min_x) * self.scale
        sy = self.rect.bottom - self.pad - (y - self.min_y) * self.scale
        return int(sx), int(sy)


def draw_track(surface: pygame.Surface, track: Track, view: WorldView) -> None:
    pygame.draw.rect(surface, GRASS, view.rect)
    left = track.left
    right = track.right
    count = len(left)
    for index in range(count):
        nxt = (index + 1) % count
        quad = [
            view.world(float(left[index, 0]), float(left[index, 1])),
            view.world(float(left[nxt, 0]), float(left[nxt, 1])),
            view.world(float(right[nxt, 0]), float(right[nxt, 1])),
            view.world(float(right[index, 0]), float(right[index, 1])),
        ]
        pygame.draw.polygon(surface, ASPHALT, quad)
    edge_left = [view.world(float(point[0]), float(point[1])) for point in left]
    edge_right = [view.world(float(point[0]), float(point[1])) for point in right]
    if len(edge_left) >= 2:
        pygame.draw.lines(surface, EDGE, True, edge_left, 2)
        pygame.draw.lines(surface, EDGE, True, edge_right, 2)
    step = max(1, count // 160)
    center = [view.world(float(track.center[i, 0]), float(track.center[i, 1])) for i in range(0, count, step * 2)]
    if len(center) >= 2:
        pygame.draw.lines(surface, CENTER, False, center, 1)
    _draw_start(surface, track, view)
    _draw_checkpoints(surface, track, view)


def _draw_start(surface: pygame.Surface, track: Track, view: WorldView) -> None:
    x0, y0, _heading = track.pose_at(0.0, -track.half_width)
    x1, y1, _heading = track.pose_at(0.0, track.half_width)
    pygame.draw.line(surface, START, view.world(x0, y0), view.world(x1, y1), 3)


def _draw_checkpoints(surface: pygame.Surface, track: Track, view: WorldView) -> None:
    for arc in track.checkpoint_s:
        if float(arc) < 8.0:
            continue
        span = track.half_width * 0.55
        x0, y0, _heading = track.pose_at(float(arc), -span)
        x1, y1, _heading = track.pose_at(float(arc), span)
        pygame.draw.line(surface, (120, 124, 132), view.world(x0, y0), view.world(x1, y1), 1)


def _sample_xy_speed(sample) -> tuple[float, float, float | None]:
    if isinstance(sample, dict):
        return float(sample["x"]), float(sample["y"]), float(sample["speed"]) if "speed" in sample else None
    if len(sample) >= 3:
        return float(sample[0]), float(sample[1]), float(sample[2])
    return float(sample[0]), float(sample[1]), None


def draw_speed_line(surface: pygame.Surface, samples: list, view: WorldView, width: int = 2) -> None:
    if len(samples) < 2:
        return
    parsed = [_sample_xy_speed(sample) for sample in samples]
    for start, end in zip(parsed, parsed[1:]):
        speed = end[2] if end[2] is not None else (start[2] if start[2] is not None else 12.0)
        pygame.draw.line(surface, speed_color(speed), view.world(start[0], start[1]), view.world(end[0], end[1]), width)


def draw_polyline(surface: pygame.Surface, points: list[tuple[float, float]], view: WorldView, color: tuple[int, int, int], width: int = 2) -> None:
    if len(points) < 2:
        return
    screen = [view.world(x, y) for x, y in points]
    pygame.draw.lines(surface, color, False, screen, width)


def draw_rays(surface: pygame.Surface, rays: list[Ray], view: WorldView) -> None:
    for ray in rays:
        color = ray_color(ray.normalized)
        thickness = 3 if ray.normalized < 0.28 else 1
        pygame.draw.line(surface, color, view.world(ray.x0, ray.y0), view.world(ray.x1, ray.y1), thickness)


def draw_car(surface: pygame.Surface, state: VehicleState, view: WorldView, color: tuple[int, int, int] = LEARNED) -> None:
    length = max(30.0, 5.2 * view.scale)
    width = max(16.0, 2.3 * view.scale)
    cx, cy = view.world(state.x, state.y)
    cos_t = math.cos(state.theta)
    sin_t = math.sin(state.theta)
    body = [(-length * 0.48, -width * 0.48), (length * 0.22, -width * 0.48), (length * 0.22, width * 0.48), (-length * 0.48, width * 0.48)]
    nose = [(length * 0.18, -width * 0.34), (length * 0.56, 0.0), (length * 0.18, width * 0.34)]

    def project(local: list[tuple[float, float]]) -> list[tuple[float, float]]:
        points = []
        for lx, ly in local:
            rx = lx * cos_t - ly * sin_t
            ry = lx * sin_t + ly * cos_t
            points.append((cx + rx, cy - ry))
        return points

    pygame.draw.polygon(surface, (12, 12, 14), project(body))
    pygame.draw.polygon(surface, color, project(body))
    pygame.draw.polygon(surface, (255, 244, 210), project(nose))
    pygame.draw.polygon(surface, (20, 20, 22), project(body), 1)


def draw_heatmap_band(surface: pygame.Surface, track: Track, hist: list[float], view: WorldView) -> None:
    if not hist or max(hist) <= 0:
        return
    peak = max(hist)
    overlay = pygame.Surface((view.rect.width, view.rect.height), pygame.SRCALPHA)
    for index, value in enumerate(hist):
        if value <= 0:
            continue
        arc = (index + 0.5) / len(hist) * track.length
        heat = value / peak
        x0, y0, _heading = track.pose_at(arc, -track.half_width * 0.92)
        x1, y1, _heading = track.pose_at(arc, track.half_width * 0.92)
        start = view.world(x0, y0)
        end = view.world(x1, y1)
        local_start = (start[0] - view.rect.left, start[1] - view.rect.top)
        local_end = (end[0] - view.rect.left, end[1] - view.rect.top)
        color = (220, int(160 * (1.0 - heat)), 48, int(70 + 160 * heat))
        pygame.draw.line(overlay, color, local_start, local_end, max(5, int(view.scale * 2.4)))
    surface.blit(overlay, view.rect.topleft)


def draw_heatmap(surface: pygame.Surface, track: Track, hist: list[float], view: WorldView) -> None:
    draw_heatmap_band(surface, track, hist, view)


def draw_markers(surface: pygame.Surface, markers: list[dict], view: WorldView, font: pygame.font.Font) -> None:
    for marker in markers:
        point = view.world(float(marker["x"]), float(marker["y"]))
        pygame.draw.circle(surface, BRAKE, point, 5)
        pygame.draw.circle(surface, (20, 20, 22), point, 5, 1)
        label = font.render(str(marker["text"]), True, TEXT)
        text_x = point[0] + 8
        if text_x + label.get_width() > view.rect.right - 6:
            text_x = point[0] - label.get_width() - 8
        surface.blit(label, (text_x, point[1] - 16))


def draw_sparkline(surface: pygame.Surface, rect: pygame.Rect, values: list[float], color: tuple[int, int, int], label: str, font: pygame.font.Font) -> None:
    pygame.draw.rect(surface, (10, 12, 16), rect)
    pygame.draw.rect(surface, STROKE, rect, 1)
    surface.blit(font.render(label, True, MUTED), (rect.x + 8, rect.y + 4))
    if len(values) < 2:
        surface.blit(font.render("no samples yet", True, MUTED), (rect.x + 8, rect.y + 28))
        return
    low = min(values)
    high = max(values)
    span = max(high - low, 1e-6)
    points = []
    for index, value in enumerate(values):
        x = rect.x + 8 + (rect.width - 16) * index / (len(values) - 1)
        y = rect.bottom - 8 - (rect.height - 28) * (value - low) / span
        points.append((int(x), int(y)))
    pygame.draw.lines(surface, color, False, points, 2)


def draw_channels(surface: pygame.Surface, rect: pygame.Rect, samples: list[dict], font: pygame.font.Font) -> None:
    """Rolling speed, throttle, brake, and steering on fixed scales."""
    specs = (
        ("speed", "SPD", 0.0, 32.0, CLASSICAL),
        ("throttle", "THR", 0.0, 1.0, THROTTLE),
        ("brake", "BRK", 0.0, 1.0, BRAKE),
        ("steer", "STR", -1.0, 1.0, GOLD),
    )
    if rect.height < 40 or not specs:
        return
    row_h = rect.height / len(specs)
    window = samples[-180:]
    for index, (key, label, low, high, color) in enumerate(specs):
        row = pygame.Rect(rect.x, int(rect.y + index * row_h), rect.width, int(row_h) - 4)
        pygame.draw.rect(surface, (10, 12, 16), row)
        pygame.draw.rect(surface, STROKE, row, 1)
        surface.blit(font.render(label, True, color), (row.x + 6, row.y + 2))
        live = window[-1].get(key, 0.0) if window else 0.0
        surface.blit(font.render(f"{float(live):+.2f}" if key == "steer" else f"{float(live):.2f}", True, TEXT), (row.x + 42, row.y + 2))
        if len(window) < 2:
            continue
        span = max(high - low, 1e-6)
        points = []
        for step, sample in enumerate(window):
            value = _clip(float(sample.get(key, 0.0)), low, high)
            x = row.x + 78 + (row.width - 88) * step / (len(window) - 1)
            y = row.bottom - 4 - (row.height - 18) * (value - low) / span
            points.append((int(x), int(y)))
        pygame.draw.lines(surface, color, False, points, 2)


def draw_bar(surface: pygame.Surface, origin: tuple[int, int], width: int, fraction: float, color: tuple[int, int, int]) -> None:
    fraction = _clip(fraction, 0.0, 1.0)
    pygame.draw.rect(surface, (32, 36, 44), pygame.Rect(origin[0], origin[1], width, 10))
    pygame.draw.rect(surface, color, pygame.Rect(origin[0], origin[1], int(width * fraction), 10))


def draw_steering(surface: pygame.Surface, origin: tuple[int, int], width: int, steering: float) -> None:
    pygame.draw.line(surface, (54, 58, 68), (origin[0], origin[1] + 6), (origin[0] + width, origin[1] + 6), 3)
    x = origin[0] + int((_clip(steering, -1.0, 1.0) + 1.0) * 0.5 * width)
    pygame.draw.circle(surface, GOLD, (x, origin[1] + 6), 6)


def draw_results_table(surface: pygame.Surface, rect: pygame.Rect, rows: list[dict], font: pygame.font.Font) -> None:
    headers = ("driver", "completion", "lap", "collisions", "off-track", "params")
    pygame.draw.rect(surface, (10, 12, 16), rect)
    if not rows:
        surface.blit(font.render("No eval JSON in data/results yet.", True, MUTED), (rect.x + 16, rect.y + 16))
        return
    col_w = rect.width / len(headers)
    for index, header in enumerate(headers):
        surface.blit(font.render(header, True, GOLD), (rect.x + 12 + int(index * col_w), rect.y + 12))
    for row_index, row in enumerate(rows):
        y = rect.y + 40 + row_index * 28
        if y > rect.bottom - 24:
            break
        lap = row.get("lap_time")
        params = row.get("parameters")
        values = (
            str(row.get("label", "")),
            f"{float(row.get('completion', 0.0)) * 100:.0f}%",
            f"{float(lap):.2f}s" if lap else "—",
            f"{float(row.get('collisions', 0.0)):.1f}",
            f"{float(row.get('off_track', 0.0)) * 100:.1f}%",
            f"{int(params)}" if params else "—",
        )
        for index, value in enumerate(values):
            surface.blit(font.render(value, True, TEXT), (rect.x + 12 + int(index * col_w), y))

