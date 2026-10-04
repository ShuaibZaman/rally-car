"""Best laps recorded from finished rollouts. Missing times stay missing."""

from __future__ import annotations

import json
from pathlib import Path

BOARD_PATH = Path("data/results/leaderboard.json")
CLASSICAL_EVAL = Path("data/results/eval_classical.json")


def load_board(path: Path = BOARD_PATH) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("entries", []))


def record_lap(driver: str, lap_time: float | None, track_seed: int, completed: bool, path: Path = BOARD_PATH) -> list[dict]:
    if not completed or lap_time is None:
        return load_board(path)
    entries = load_board(path)
    entries.append(
        {
            "driver": driver,
            "lap_time": float(lap_time),
            "track_seed": int(track_seed),
            "completed": True,
        }
    )
    best: dict[tuple[str, int], dict] = {}
    for entry in entries:
        key = (entry["driver"], int(entry["track_seed"]))
        current = best.get(key)
        if current is None or entry["lap_time"] < current["lap_time"]:
            best[key] = entry
    ranked = sorted(best.values(), key=lambda item: (item["track_seed"], item["lap_time"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": ranked}, indent=2), encoding="utf-8")
    return ranked


def reference_lap(track_seed: int, path: Path = CLASSICAL_EVAL) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    times = [
        float(episode["lap_time"])
        for episode in payload.get("episodes", [])
        if episode.get("completed") and episode.get("lap_time") and int(episode.get("track_seed", -1)) == int(track_seed)
    ]
    if not times:
        return None
    return min(times)
