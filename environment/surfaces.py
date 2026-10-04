"""Surfaces and weather scale grip and acceleration. Default is dry asphalt."""

from __future__ import annotations

SURFACES: dict[str, dict[str, float]] = {
    "asphalt": {"grip": 1.0, "accel": 1.0},
    "gravel": {"grip": 0.65, "accel": 0.75},
    "mud": {"grip": 0.40, "accel": 0.45},
    "snow": {"grip": 0.28, "accel": 0.40},
    "dirt": {"grip": 0.55, "accel": 0.70},
}

WEATHER: dict[str, dict[str, float]] = {
    "dry": {"grip": 1.0, "drag": 1.0},
    "rain": {"grip": 0.70, "drag": 1.05},
    "snow": {"grip": 0.45, "drag": 1.10},
}


def combined_surface(surface: str, weather: str) -> dict[str, float]:
    if surface not in SURFACES:
        raise KeyError(surface)
    if weather not in WEATHER:
        raise KeyError(weather)
    surf = SURFACES[surface]
    sky = WEATHER[weather]
    return {
        "grip": surf["grip"] * sky["grip"],
        "accel": surf["accel"],
        "drag": sky["drag"],
    }
