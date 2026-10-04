"""Reward terms are not training losses.

Variants A–D change the objective. Optimizer and architecture live elsewhere.
Variant A is the progress-only reward-hacking fixture: leaving the track does
not end the episode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from environment.types import RewardBreakdown, RewardWeights

REWARD_INDEX = {"A": 1, "B": 2, "C": 3, "D": 4}

VARIANTS: dict[str, RewardWeights] = {
    "A": RewardWeights(
        progress=1.0,
        terminate_collision=False,
        terminate_off_track=False,
    ),
    "B": RewardWeights(
        progress=1.0,
        collision=10.0,
        terminate_collision=True,
    ),
    "C": RewardWeights(
        progress=1.0,
        collision=10.0,
        time=0.4,
        off_track=1.5,
        lap_bonus=15.0,
        terminate_collision=True,
    ),
    "D": RewardWeights(
        progress=1.0,
        collision=10.0,
        time=0.4,
        off_track=1.5,
        lap_bonus=15.0,
        racing_line=0.35,
        centering=0.05,
        speed=0.02,
        steering=0.05,
        braking=0.02,
        terminate_collision=True,
    ),
}


def get_weights(name: str) -> RewardWeights:
    if name not in VARIANTS:
        raise KeyError(name)
    return VARIANTS[name]


def compute_reward(
    weights: RewardWeights,
    *,
    ds: float,
    dt: float,
    off_track: bool,
    collision: bool,
    steering: float,
    brake: float,
    lateral: float,
    half_width: float,
    curvature: float,
    speed: float,
    lap_completed: bool,
) -> RewardBreakdown:
    ideal = 0.0
    if half_width > 0.0:
        ideal = -float(np.sign(curvature)) * half_width * float(np.clip(abs(curvature) / 0.04, 0.0, 0.65))
    line_error = abs(lateral - ideal) / max(half_width, 1e-6)
    center_error = abs(lateral) / max(half_width, 1e-6)
    return RewardBreakdown(
        progress=weights.progress * max(ds, 0.0),
        speed=weights.speed * speed * dt,
        centering=-weights.centering * center_error,
        off_track=(-weights.off_track * dt) if off_track else 0.0,
        steering=-weights.steering * abs(steering) * dt,
        braking=-weights.braking * brake * dt,
        collision=(-weights.collision) if collision else 0.0,
        time=-weights.time * dt,
        lap_bonus=weights.lap_bonus if lap_completed else 0.0,
        racing_line=-weights.racing_line * line_error,
    )


def write_reward_configs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    notes = {
        "A": "Progress only. Off-track driving still scores. Reward-hacking fixture.",
        "B": "Progress minus a collision penalty.",
        "C": "Racing: progress, collision, time, off-track, plus a lap bonus.",
        "D": "Racing plus a racing-line term toward the inside of curves.",
    }
    for name, weights in VARIANTS.items():
        payload = {
            "reward": name,
            "index": REWARD_INDEX[name],
            "notes": notes[name],
            "weights": weights.as_dict(),
        }
        (directory / f"reward_{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
