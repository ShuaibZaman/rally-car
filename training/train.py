"""Short PPO entry point.

The default budget is a smoke run. It writes a checkpoint bundle and metrics.
It does not claim the car has learned to race.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.env import RallyEnv
from environment.types import EnvConfig
from training.algorithms import build_model
from training.experiment import ExperimentConfig


class HistoryCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.loss_history: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode:
                self.episode_rewards.append(float(episode["r"]))
                self.episode_lengths.append(int(episode["l"]))
        return True

    def _on_rollout_start(self) -> None:
        self.capture_loss()

    def capture_loss(self) -> None:
        values = getattr(getattr(self.model, "logger", None), "name_to_value", {})
        if "train/loss" in values:
            loss = float(values["train/loss"])
            if not self.loss_history or self.loss_history[-1] != loss:
                self.loss_history.append(loss)


def _loss_snapshot(model) -> dict[str, float]:
    values = getattr(getattr(model, "logger", None), "name_to_value", {})
    return {key: float(value) for key, value in values.items() if "loss" in key}


def train(experiment: ExperimentConfig, episode_steps: int, device: str, checkpoint_dir: Path) -> dict:
    if experiment.observation == "vision":
        raise RuntimeError("Vision training is scaffolded. Pass --allow-vision only after you mean to run it.")
    env_config = EnvConfig(
        preset=experiment.preset,
        track_seed=experiment.track_seed,
        reward=experiment.reward,
        max_steps=episode_steps,
        observation=experiment.observation,
    )

    def _factory():
        return Monitor(RallyEnv(env_config))

    vec = DummyVecEnv([_factory])
    n_steps = min(1024, max(64, experiment.max_steps))
    model = build_model(
        experiment.algorithm,
        vec,
        experiment.learning_rate,
        experiment.hidden,
        experiment.train_seed,
        device,
        experiment.entropy_coef,
        n_steps=n_steps,
    )
    history = HistoryCallback()
    model.learn(total_timesteps=experiment.max_steps, callback=history, progress_bar=False)
    history.capture_loss()
    parameters = int(sum(param.numel() for param in model.policy.parameters()))
    losses = _loss_snapshot(model)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    zip_path = checkpoint_dir / "ppo_smoke.zip"
    model.save(str(checkpoint_dir / "ppo_smoke"))
    bundle = {
        "experiment": experiment.as_dict(),
        "parameters": parameters,
        "episode_rewards": history.episode_rewards,
        "episode_lengths": history.episode_lengths,
        "loss_history": history.loss_history,
        "losses": losses,
        "device": device,
        "zip": str(zip_path),
    }
    step_name = f"episode_{experiment.max_steps:06d}.pt"
    torch.save({"policy": model.policy.state_dict(), **bundle}, checkpoint_dir / step_name)
    (checkpoint_dir / "ppo_smoke.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    vec.close()
    return bundle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-train a sensor PPO driver.")
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--preset", default="easy")
    parser.add_argument("--track-seed", type=int, default=1_000_000)
    parser.add_argument("--reward", default="C", choices=("A", "B", "C", "D"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--observation", default="sensors", choices=("sensors", "vision"))
    parser.add_argument("--allow-vision", action="store_true")
    parser.add_argument("--algorithm", default="PPO")
    parser.add_argument("--checkpoint-dir", default="training/checkpoints")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.observation == "vision" and not args.allow_vision:
        raise SystemExit("Vision training is scaffolded. Re-run with --allow-vision if you intend to train it.")
    from agents.policies.mlp import trunk_parameter_count
    from environment.sensors import OBS_DIM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment = ExperimentConfig(
        algorithm=args.algorithm,
        architecture="MLP2",
        hidden=[128, 128],
        observation=args.observation,
        reward=args.reward,
        optimizer="Adam",
        learning_rate=args.learning_rate,
        train_seed=args.seed,
        env_seed=args.seed,
        track_seed=args.track_seed,
        preset=args.preset,
        max_steps=args.max_steps,
        notes="Smoke run. A few thousand steps is not a trained racing policy.",
        trunk_parameters=trunk_parameter_count(OBS_DIM, [128, 128], 3),
    )
    bundle = train(experiment, args.episode_steps, device, Path(args.checkpoint_dir))
    results = Path("data/results")
    results.mkdir(parents=True, exist_ok=True)
    (results / "ppo_smoke_train.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(experiment.experiment_id)
    print(f"device={device} parameters={bundle['parameters']} episodes={len(bundle['episode_rewards'])}")
    print(f"checkpoint={args.checkpoint_dir}")


if __name__ == "__main__":
    main()
