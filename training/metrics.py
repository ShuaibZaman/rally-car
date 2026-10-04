"""Episode metrics and the composite rally score.

The score is always returned together with the terms that produced it.
"""

from __future__ import annotations

import math
from statistics import mean, pstdev

from environment.types import EpisodeMetrics


def rally_score(metrics: EpisodeMetrics) -> dict:
    completion = 1.0 if metrics.completed else float(metrics.progress)
    speed_term = float(max(0.0, min(1.5, metrics.avg_speed / 25.0)))
    smoothness_term = float(max(0.0, min(1.0, 1.0 - metrics.steering_smoothness / 0.5)))
    crash_term = float(metrics.collisions)
    off_track_term = float(metrics.off_track_fraction)
    score = completion + 0.5 * speed_term + 0.25 * smoothness_term - crash_term - off_track_term
    return {
        "rally_score": score,
        "completion_term": completion,
        "speed_term": speed_term,
        "smoothness_term": smoothness_term,
        "crash_term": crash_term,
        "off_track_term": off_track_term,
        **metrics.as_dict(),
    }


def aggregate_metrics(rows: list[dict]) -> dict:
    """Mean and population spread across episodes or seeds."""
    if not rows:
        raise ValueError("no rows to aggregate")
    keys = [
        "progress",
        "collisions",
        "avg_speed",
        "off_track_fraction",
        "centerline_deviation",
        "steering_smoothness",
        "throttle_efficiency",
        "recoveries",
        "distance",
        "fuel",
        "rally_score",
    ]
    summary: dict = {"n": len(rows), "completed_rate": mean(1.0 if row.get("completed") else 0.0 for row in rows)}
    lap_times = [row["lap_time"] for row in rows if row.get("lap_time")]
    summary["lap_time"] = _spread(lap_times) if lap_times else {"mean": None, "std": None, "n": 0}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and row[key] is not None]
        summary[key] = _spread(values)
    return summary


def _spread(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def crash_histogram(samples: list[dict], length: float, bins: int = 40) -> list[float]:
    hist = [0.0] * bins
    if length <= 0:
        return hist
    for sample in samples:
        if sample.get("collision") or sample.get("off_track"):
            index = int(sample["s"] / length * bins) % bins
            hist[index] += 1.0
    return hist


def describe_trace(samples: list[dict]) -> list[str]:
    """Qualitative notes that the trace actually supports."""
    notes: list[str] = []
    if not samples:
        return ["No trace recorded."]
    corners = [sample for sample in samples if sample.get("curvature_ahead", 0.0) > 0.02]
    if len(corners) >= 8:
        brake = sum(sample["brake"] for sample in corners) / len(corners)
        if brake > 0.25:
            notes.append("Brakes before corners.")
        elif brake < 0.05:
            notes.append("Does not brake before corners.")
    exits = [sample for sample in samples if sample.get("curvature_ahead", 1.0) < 0.005 and sample.get("speed", 0.0) > 0]
    if len(exits) >= 8:
        exit_speed = sum(sample["speed"] for sample in exits) / len(exits)
        corner_speed = (
            sum(sample["speed"] for sample in corners) / len(corners) if corners else exit_speed
        )
        if exit_speed > corner_speed + 2.0:
            notes.append("Accelerates on corner exit.")
        elif exit_speed + 1.0 < corner_speed:
            notes.append("Exits corners slower than it enters them.")
    steer = [sample["steer"] for sample in samples]
    if len(steer) > 2:
        deltas = [abs(b - a) for a, b in zip(steer, steer[1:])]
        if sum(deltas) / len(deltas) < 0.04:
            notes.append("Steering stays smooth.")
    if any(sample.get("collision") for sample in samples):
        notes.append("The trace includes a collision.")
    if not notes:
        notes.append("No strong qualitative pattern in this trace.")
    return notes


def corner_markers(samples: list[dict], limit: int = 3) -> list[dict]:
    """Map positions for notes the trace actually supports. Skip samples with no coordinates."""
    notes = set(describe_trace(samples))
    located = [sample for sample in samples if "x" in sample and "y" in sample]
    if not located:
        return []
    markers: list[dict] = []

    def spaced(candidates: list[dict], text: str) -> None:
        last_s = -1e9
        for sample in candidates:
            arc = float(sample.get("s", sample["x"]))
            if abs(arc - last_s) < 140.0:
                continue
            markers.append({"x": float(sample["x"]), "y": float(sample["y"]), "text": text})
            last_s = arc
            if len(markers) >= limit:
                return

    if "Brakes before corners." in notes:
        spaced([sample for sample in located if sample.get("curvature_ahead", 0.0) > 0.02 and sample.get("brake", 0.0) > 0.25], "brake")
    elif "Does not brake before corners." in notes:
        spaced([sample for sample in located if sample.get("curvature_ahead", 0.0) > 0.02 and sample.get("brake", 0.0) < 0.05], "no brake")
    if len(markers) < limit and "The trace includes a collision." in notes:
        spaced([sample for sample in located if sample.get("collision")], "crash")
    return markers[:limit]
