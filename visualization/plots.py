"""Matplotlib figures for the policy slice and architecture comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from environment.sensors import neutral_observation


def save_policy_slice(predict, path: Path, speed_steps: int = 21, curvature_steps: int = 21) -> None:
    speeds = np.linspace(0.0, 32.0, speed_steps)
    curvatures = np.linspace(0.0, 0.08, curvature_steps)
    brake = np.zeros((speed_steps, curvature_steps), dtype=np.float32)
    steer = np.zeros_like(brake)
    for i, speed in enumerate(speeds):
        for j, curvature in enumerate(curvatures):
            action = np.asarray(predict(neutral_observation(float(speed), float(curvature))), dtype=np.float32)
            steer[i, j] = action[0]
            brake[i, j] = action[2]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, grid, title in (
        (axes[0], brake, "Brake"),
        (axes[1], steer, "Steering"),
    ):
        image = axis.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[float(curvatures[0]), float(curvatures[-1]), float(speeds[0]), float(speeds[-1])],
        )
        axis.set_xlabel("curvature (1/m)")
        axis.set_ylabel("speed (m/s)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_comparison_bars(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["label"] for row in rows]
    completion = [float(row.get("completion", 0.0)) for row in rows]
    figure, axis = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    axis.barh(labels, completion, color="#d4b44a")
    axis.set_xlim(0, 1)
    axis.set_xlabel("completion")
    axis.set_title("Model comparison")
    figure.savefig(path, dpi=120)
    plt.close(figure)
