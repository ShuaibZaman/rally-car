"""Named experiment configs. None of these are launched as a sweep."""

from __future__ import annotations

import json
from pathlib import Path

from agents.policies.mlp import trunk_parameter_count
from environment.sensors import OBS_DIM
from training.experiment import ExperimentConfig

DEPTH_HIDDEN = {
    "depth2": [78, 78],
    "depth4": [48, 48, 48, 48],
    "depth6": [38, 38, 38, 38, 38, 38],
}

CAPACITY_HIDDEN = {
    "cap_10k": [96, 96],
    "cap_50k": [220, 220],
    "cap_100k": [310, 310],
    "cap_500k": [700, 700],
    "cap_1m": [990, 990],
}

ACTION_DIM = 3


def _config(name: str, **kwargs) -> dict:
    hidden = list(kwargs.get("hidden", [128, 128]))
    cfg = ExperimentConfig(hidden=hidden, **{k: v for k, v in kwargs.items() if k != "hidden"})
    data = cfg.as_dict()
    data["name"] = name
    data["trunk_parameters"] = trunk_parameter_count(OBS_DIM, hidden, ACTION_DIM)
    data["train"] = False
    return data


def all_configs() -> list[dict]:
    configs = [
        _config("ppo_mlp2_smoke", algorithm="PPO", architecture="MLP2", hidden=[128, 128], reward="C", notes="Short smoke, not a finished driver."),
    ]
    for name, hidden in DEPTH_HIDDEN.items():
        configs.append(
            _config(
                name,
                algorithm="PPO",
                architecture=name,
                hidden=hidden,
                reward="C",
                notes="Depth comparison at approximately fixed trunk size.",
            )
        )
    for name, hidden in CAPACITY_HIDDEN.items():
        configs.append(
            _config(
                name,
                algorithm="PPO",
                architecture=name,
                hidden=hidden,
                reward="C",
                notes="Capacity ladder. Defined, not trained.",
            )
        )
    for reward in ("A", "B", "C", "D"):
        configs.append(
            _config(
                f"reward_{reward}",
                algorithm="PPO",
                architecture="MLP2",
                hidden=[128, 128],
                reward=reward,
                notes="Reward comparison. Same network budget.",
            )
        )
    for algo in ("PPO", "SAC", "TD3"):
        configs.append(
            _config(
                f"algo_{algo.lower()}",
                algorithm=algo,
                architecture="MLP2",
                hidden=[128, 128],
                reward="C",
                notes="Algorithm comparison at matched hidden size.",
            )
        )
    configs.append(
        _config(
            "memory_lstm",
            algorithm="LSTM",
            architecture="MLP_LSTM",
            hidden=[128, 128],
            reward="C",
            notes="Recurrent policy via sb3-contrib. Not part of the smoke run.",
        )
    )
    configs.append(
        _config(
            "vision_cnn",
            algorithm="PPO",
            architecture="CNN",
            hidden=[128],
            observation="vision",
            reward="C",
            notes="Pixel observation. Refused by train.py unless --allow-vision is passed.",
        )
    )
    configs.append(
        _config(
            "generalization_holdout",
            algorithm="PPO",
            architecture="MLP2",
            hidden=[128, 128],
            reward="C",
            split="test",
            track_seed=3_000_000,
            notes="Evaluate on the test pool. Do not train on these seeds.",
        )
    )
    return configs


def write_configs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for config in all_configs():
        path = root / f"{config['name']}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
