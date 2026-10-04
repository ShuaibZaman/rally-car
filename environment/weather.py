"""Weather is a grip and drag modifier on top of the surface table."""

from __future__ import annotations

from environment.surfaces import WEATHER, combined_surface


def weather_names() -> list[str]:
    return list(WEATHER)


def apply_weather(surface: str, weather: str) -> dict[str, float]:
    return combined_surface(surface, weather)
