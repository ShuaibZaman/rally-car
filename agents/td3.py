"""TD3 builder. Training runs only when explicitly requested."""

from __future__ import annotations


def build_td3(env, learning_rate: float, hidden: list[int], seed: int, device: str):
    from stable_baselines3 import TD3

    return TD3(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        policy_kwargs={"net_arch": hidden},
        seed=seed,
        device=device,
        verbose=0,
    )
