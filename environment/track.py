"""Closed-curve track: centerline, boundaries, curvature, checkpoints."""

from __future__ import annotations

import math

import numpy as np

from environment.types import Projection, TrackMeta


def _count_turns(curvature: np.ndarray, seg_len: np.ndarray, min_abs: float = 0.008, min_len: float = 12.0) -> int:
    signs = np.zeros(len(curvature), dtype=np.int8)
    signs[curvature > min_abs] = 1
    signs[curvature < -min_abs] = -1
    turns = 0
    prev = 0
    run = 0.0
    for sign, length in zip(signs, seg_len):
        sign = int(sign)
        if sign == 0:
            if prev != 0 and run >= min_len:
                turns += 1
            prev = 0
            run = 0.0
            continue
        if sign == prev:
            run += float(length)
        else:
            if prev != 0 and run >= min_len:
                turns += 1
            prev = sign
            run = float(length)
    if prev != 0 and run >= min_len:
        turns += 1
    return int(turns)


class Track:
    def __init__(
        self,
        center: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        s: np.ndarray,
        seg_len: np.ndarray,
        heading: np.ndarray,
        curvature: np.ndarray,
        checkpoint_s: np.ndarray,
        meta: TrackMeta,
    ) -> None:
        self.center = center
        self.left = left
        self.right = right
        self.s = s
        self.seg_len = seg_len
        self.heading = heading
        self.curvature = curvature
        self.checkpoint_s = checkpoint_s
        self.meta = meta
        self.width = float(meta.width)
        self.half_width = self.width * 0.5
        self.length = float(meta.length_m)

    @classmethod
    def from_centerline(
        cls,
        points: np.ndarray,
        width: float,
        seed: int,
        difficulty: float,
        difficulty_name: str,
        checkpoint_spacing: float = 25.0,
    ) -> Track:
        center = np.asarray(points, dtype=np.float64)
        if len(center) < 8:
            raise ValueError("centerline needs at least 8 points")
        nxt = np.roll(center, -1, axis=0)
        seg = nxt - center
        seg_len = np.linalg.norm(seg, axis=1)
        seg_len = np.maximum(seg_len, 1e-6)
        heading = np.arctan2(seg[:, 1], seg[:, 0])
        next_heading = np.roll(heading, -1)
        dh = np.arctan2(np.sin(next_heading - heading), np.cos(next_heading - heading))
        curvature = dh / seg_len
        normals = np.stack([-np.sin(heading), np.cos(heading)], axis=1)
        half = width * 0.5
        left = center + normals * half
        right = center - normals * half
        s = np.cumsum(np.concatenate([[0.0], seg_len[:-1]]))
        length = float(np.sum(seg_len))
        spacing = min(checkpoint_spacing, length / 4.0)
        count = max(4, int(math.floor(length / spacing)))
        checkpoint_s = np.linspace(0.0, length, count, endpoint=False)
        meta = TrackMeta(
            track_id=int(seed),
            seed=int(seed),
            difficulty=float(difficulty),
            difficulty_name=difficulty_name,
            turns=_count_turns(curvature, seg_len),
            length_m=length,
            width=float(width),
        )
        return cls(center, left, right, s, seg_len, heading, curvature, checkpoint_s, meta)

    def project(self, x: float, y: float, hint: int | None = None) -> Projection:
        point = np.array([x, y], dtype=np.float64)
        n = len(self.center)
        if hint is None:
            candidates = range(n)
        else:
            window = 50
            candidates = [(hint + k) % n for k in range(-window, window + 1)]
        best_d = 1e18
        best_i = 0
        best_t = 0.0
        best_q = self.center[0]
        for i in candidates:
            a = self.center[i]
            b = self.center[(i + 1) % n]
            ab = b - a
            length2 = float(np.dot(ab, ab))
            if length2 < 1e-12:
                continue
            t = float(np.clip(np.dot(point - a, ab) / length2, 0.0, 1.0))
            q = a + t * ab
            dist2 = float(np.dot(point - q, point - q))
            if dist2 < best_d:
                best_d = dist2
                best_i = i
                best_t = t
                best_q = q
        ab = self.center[(best_i + 1) % n] - self.center[best_i]
        ab_norm = float(np.linalg.norm(ab))
        cross = float(ab[0] * (point[1] - best_q[1]) - ab[1] * (point[0] - best_q[0]))
        lateral = cross / max(ab_norm, 1e-9)
        heading = float(math.atan2(ab[1], ab[0]))
        s = float(self.s[best_i] + best_t * self.seg_len[best_i])
        if s >= self.length:
            s -= self.length
        return Projection(s=s, lateral=float(lateral), heading=heading, curvature=float(self.curvature[best_i]), index=best_i)

    def pose_at(self, s: float, lateral: float = 0.0) -> tuple[float, float, float]:
        n = len(self.center)
        s = float(s % self.length)
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        i = max(0, min(n - 1, i))
        span = float(self.seg_len[i])
        t = 0.0 if span <= 1e-9 else float(np.clip((s - self.s[i]) / span, 0.0, 1.0))
        a = self.center[i]
        b = self.center[(i + 1) % n]
        pos = a * (1.0 - t) + b * t
        heading = float(self.heading[i])
        pos = pos + np.array([-math.sin(heading), math.cos(heading)]) * lateral
        return float(pos[0]), float(pos[1]), heading

    def curvature_at(self, s: float) -> float:
        s = float(s % self.length)
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        i = max(0, min(len(self.curvature) - 1, i))
        return float(self.curvature[i])

    def max_curvature_ahead(self, s: float, distance: float, samples: int = 8) -> float:
        if distance <= 0.0:
            return abs(self.curvature_at(s))
        values = [abs(self.curvature_at(s + distance * i / samples)) for i in range(samples + 1)]
        return float(max(values))

    def on_track(self, lateral: float, margin: float = 0.0) -> bool:
        return abs(lateral) <= self.half_width - margin
