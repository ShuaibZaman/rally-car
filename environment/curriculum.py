"""Curriculum stages. The trainer does not auto-advance."""

from __future__ import annotations

STAGES: list[dict] = [
    {
        "stage": 1,
        "name": "straight",
        "preset": "straight",
        "goal": "Learn throttle and basic steering.",
        "randomize_start": False,
        "lateral_force": 0.0,
        "surface": "asphalt",
    },
    {
        "stage": 2,
        "name": "gentle",
        "preset": "easy",
        "goal": "Learn turning.",
        "randomize_start": False,
        "lateral_force": 0.0,
        "surface": "asphalt",
    },
    {
        "stage": 3,
        "name": "sharp",
        "preset": "medium",
        "goal": "Learn braking and corner entry.",
        "randomize_start": False,
        "lateral_force": 0.0,
        "surface": "asphalt",
    },
    {
        "stage": 4,
        "name": "s_curves",
        "preset": "hard",
        "goal": "Learn rapid direction changes.",
        "randomize_start": False,
        "lateral_force": 0.0,
        "surface": "asphalt",
    },
    {
        "stage": 5,
        "name": "hairpins",
        "preset": "extreme",
        "goal": "Learn aggressive braking and acceleration.",
        "randomize_start": False,
        "lateral_force": 0.0,
        "surface": "asphalt",
    },
    {
        "stage": 6,
        "name": "random",
        "preset": "hard",
        "goal": "Generalize across seeds.",
        "randomize_start": True,
        "lateral_force": 0.0,
        "surface": "asphalt",
    },
    {
        "stage": 7,
        "name": "disturbances",
        "preset": "medium",
        "goal": "Recover from grip loss and a lateral push.",
        "randomize_start": True,
        "lateral_force": 2.5,
        "surface": "gravel",
    },
]


def stage_config(stage: int) -> dict:
    for item in STAGES:
        if item["stage"] == stage:
            return dict(item)
    raise KeyError(stage)
