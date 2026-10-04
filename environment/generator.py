"""Seeded closed tracks and the train / validation / test pools.

Easy tracks are stadiums. Harder presets are radial closed curves. The same
(seed, preset) pair always rebuilds the same centerline.
"""

from __future__ import annotations

import math

import numpy as np

from environment.track import Track

TRAIN_SEED_START = 1_000_000
VAL_SEED_START = 2_000_000
TEST_SEED_START = 3_000_000
SPLIT_SIZES = {"train": 1000, "val": 100, "test": 100}
SPLIT_START = {"train": TRAIN_SEED_START, "val": VAL_SEED_START, "test": TEST_SEED_START}

EASY_FIXTURE_SEED = TRAIN_SEED_START
TEST_FIXTURE_SEED = TEST_SEED_START

PRESETS: dict[str, dict] = {
    "straight": {
        "kind": "stadium",
        "straight": 280.0,
        "radius": 90.0,
        "width": 16.0,
        "jitter": 0.0,
        "difficulty": 0.05,
        "samples": 520,
    },
    "easy": {
        "kind": "stadium",
        "straight": 100.0,
        "radius": 48.0,
        "width": 16.0,
        "jitter": 8.0,
        "difficulty": 0.20,
        "samples": 480,
    },
    "medium": {
        "kind": "radial",
        "radius": 90.0,
        "harmonics": 3,
        "amp": 0.16,
        "width": 13.0,
        "samples": 520,
        "difficulty": 0.45,
    },
    "hard": {
        "kind": "radial",
        "radius": 78.0,
        "harmonics": 5,
        "amp": 0.26,
        "width": 11.0,
        "samples": 600,
        "difficulty": 0.70,
    },
    "extreme": {
        "kind": "radial",
        "radius": 68.0,
        "harmonics": 7,
        "amp": 0.34,
        "width": 9.0,
        "samples": 680,
        "difficulty": 0.90,
    },
}


def split_seeds(split: str, limit: int | None = None) -> list[int]:
    if split not in SPLIT_START:
        raise KeyError(split)
    start = SPLIT_START[split]
    count = SPLIT_SIZES[split]
    seeds = list(range(start, start + count))
    if limit is not None:
        return seeds[:limit]
    return seeds


def _stadium_points(straight: float, radius: float, samples: int) -> np.ndarray:
    perimeter = 2.0 * straight + 2.0 * math.pi * radius
    ds = perimeter / samples
    points = np.zeros((samples, 2), dtype=np.float64)
    for i in range(samples):
        points[i] = _stadium_point(i * ds, straight, radius)
    return points


def _stadium_point(distance: float, straight: float, radius: float) -> tuple[float, float]:
    cap = math.pi * radius
    if distance < straight:
        return (-straight * 0.5 + distance, -radius)
    distance -= straight
    if distance < cap:
        angle = -math.pi * 0.5 + distance / radius
        return (straight * 0.5 + radius * math.cos(angle), radius * math.sin(angle))
    distance -= cap
    if distance < straight:
        return (straight * 0.5 - distance, radius)
    distance -= straight
    angle = math.pi * 0.5 + distance / radius
    return (-straight * 0.5 + radius * math.cos(angle), radius * math.sin(angle))


def _radial_points(seed: int, preset: dict) -> np.ndarray:
    samples = int(preset["samples"])
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    amp_scale = 1.0
    points = None
    for attempt in range(8):
        rng = np.random.default_rng(seed + attempt * 17)
        radius = np.full(samples, preset["radius"], dtype=np.float64)
        for k in range(1, int(preset["harmonics"]) + 1):
            amp = float(rng.uniform(0.02, preset["amp"])) / k * amp_scale
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            radius = radius + preset["radius"] * amp * np.sin(k * theta + phase)
        radius = np.clip(radius, preset["radius"] * 0.4, preset["radius"] * 1.7)
        points = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
        if not _too_close(points, preset["width"]):
            return _resample(points, spacing=2.0)
        amp_scale *= 0.72
    return _resample(points, spacing=2.0)


def _too_close(points: np.ndarray, width: float) -> bool:
    sampled = points[::3]
    n = len(sampled)
    if n < 8:
        return False
    delta = sampled[:, None, :] - sampled[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    index = np.arange(n)
    adjacent = np.abs((index[:, None] - index[None, :] + n) % n) < max(8, n // 12)
    dist = np.where(adjacent, np.inf, dist)
    np.fill_diagonal(dist, np.inf)
    return bool(np.min(dist) < width * 1.15)


def _resample(points: np.ndarray, spacing: float) -> np.ndarray:
    closed = np.vstack([points, points[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    count = max(64, int(total / spacing))
    targets = np.linspace(0.0, total, count, endpoint=False)
    out = np.zeros((count, 2), dtype=np.float64)
    j = 0
    for i, target in enumerate(targets):
        while j < len(seg) - 1 and s[j + 1] < target:
            j += 1
        local = 0.0 if seg[j] <= 1e-9 else (target - s[j]) / seg[j]
        local = float(np.clip(local, 0.0, 1.0))
        out[i] = closed[j] * (1.0 - local) + closed[j + 1] * local
    return out


def generate_track(seed: int, preset: str = "easy") -> Track:
    if preset not in PRESETS:
        raise KeyError(preset)
    spec = PRESETS[preset]
    rng = np.random.default_rng(int(seed))
    difficulty = float(np.clip(spec["difficulty"] + rng.uniform(-0.02, 0.02), 0.0, 1.0))
    if spec["kind"] == "stadium":
        jitter = float(spec["jitter"])
        straight = spec["straight"] + float(rng.uniform(-jitter, jitter))
        radius = spec["radius"] + float(rng.uniform(-jitter * 0.4, jitter * 0.4))
        points = _stadium_points(straight, radius, int(spec["samples"]))
        width = float(spec["width"])
    else:
        points = _radial_points(int(seed), spec)
        width = float(spec["width"])
    return Track.from_centerline(points, width, int(seed), difficulty, preset)
