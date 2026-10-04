"""Experiment identity. Reward variants and optimizers stay in separate fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from environment.rewards import REWARD_INDEX


@dataclass
class ExperimentConfig:
    algorithm: str = "PPO"
    architecture: str = "MLP2"
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    observation: str = "sensors"
    reward: str = "C"
    optimizer: str = "Adam"
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    train_seed: int = 42
    env_seed: int = 42
    track_seed: int = 1_000_000
    split: str = "train"
    preset: str = "easy"
    max_steps: int = 4096
    trunk_parameters: int | None = None
    notes: str = ""

    @property
    def experiment_id(self) -> str:
        return make_experiment_id(self.algorithm, self.architecture, self.reward, self.train_seed)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["experiment_id"] = self.experiment_id
        return data


def make_experiment_id(algorithm: str, architecture: str, reward: str, train_seed: int) -> str:
    if reward not in REWARD_INDEX:
        raise KeyError(reward)
    return f"{algorithm}_{architecture}_REWARD{REWARD_INDEX[reward]}_SEED{int(train_seed)}"
