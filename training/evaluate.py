"""Score a checkpoint, or the classical driver, on one or more track seeds.

Pass several seeds to get a mean and a spread. The default is one short episode
so this stays a tool rather than a study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agents.classical import PurePursuit
from environment.env import RallyEnv
from environment.generator import split_seeds
from environment.types import EnvConfig
from training.metrics import aggregate_metrics, rally_score


def _row_from_env(env, seed: int) -> dict:
    row = rally_score(env.episode_metrics())
    row["track_seed"] = int(seed)
    row["preset"] = env.config.preset
    return row


def evaluate_policy(predict, seeds: list[int], episodes: int, preset: str, reward: str, episode_steps: int) -> tuple[list[dict], dict]:
    rows = []
    for seed in seeds:
        for episode in range(episodes):
            env = RallyEnv(
                EnvConfig(preset=preset, track_seed=int(seed), reward=reward, max_steps=episode_steps)
            )
            obs, _info = env.reset(seed=episode)
            done = False
            while not done:
                action = predict(obs, env)
                obs, _reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated
            rows.append(_row_from_env(env, seed))
            env.close()
    return rows, aggregate_metrics(rows)


def evaluate_classical(seeds: list[int], episodes: int, preset: str, reward: str, episode_steps: int) -> tuple[list[dict], dict]:
    expert = PurePursuit()

    def predict(_obs, env):
        return expert.act(env)

    return evaluate_policy(predict, seeds, episodes, preset, reward, episode_steps)


def evaluate_checkpoint(path: Path, seeds: list[int], episodes: int, preset: str, reward: str, episode_steps: int) -> tuple[list[dict], dict]:
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")

    def predict(obs, _env):
        action, _state = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return evaluate_policy(predict, seeds, episodes, preset, reward, episode_steps)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a rally policy on held-out seeds.")
    parser.add_argument("--checkpoint", default="training/checkpoints/ppo_smoke.zip")
    parser.add_argument("--classical", action="store_true")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed-limit", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--preset", default="easy")
    parser.add_argument("--reward", default="C")
    parser.add_argument("--episode-steps", type=int, default=1500)
    parser.add_argument("--out", default="")
    return parser.parse_args(argv)


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(part) for part in args.seeds.split(",") if part]
    return split_seeds(args.split, limit=args.seed_limit)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = resolve_seeds(args)
    parameters = None
    if args.classical:
        rows, summary = evaluate_classical(seeds, args.episodes, args.preset, args.reward, args.episode_steps)
        label = "classical"
    else:
        from stable_baselines3 import PPO

        loaded = PPO.load(args.checkpoint, device="cpu")
        parameters = int(sum(param.numel() for param in loaded.policy.parameters()))
        rows, summary = evaluate_checkpoint(
            Path(args.checkpoint), seeds, args.episodes, args.preset, args.reward, args.episode_steps
        )
        label = Path(args.out).stem.removeprefix("eval_") if args.out else Path(args.checkpoint).stem
    payload = {"label": label, "seeds": seeds, "episodes": rows, "summary": summary, "parameters": parameters}
    out = Path(args.out) if args.out else Path("data/results") / f"eval_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(
        "completion_rate={completed:.2f} progress={progress:.2f} collisions={collisions:.2f}".format(
            completed=summary["completed_rate"],
            progress=summary["progress"]["mean"],
            collisions=summary["collisions"]["mean"],
        )
    )


if __name__ == "__main__":
    main()
