"""Algorithm dispatch shared by train.py. PPO is the only smoke default."""

from __future__ import annotations

from agents.sac import build_recurrent_ppo, build_sac
from agents.td3 import build_td3


def ppo_policy_kwargs(hidden: list[int]) -> dict:
    return {"net_arch": {"pi": list(hidden), "vf": list(hidden)}}


def build_model(
    algorithm: str,
    env,
    learning_rate: float,
    hidden: list[int],
    seed: int,
    device: str,
    entropy_coef: float,
    n_steps: int = 1024,
):
    name = algorithm.upper()
    if name == "PPO":
        from stable_baselines3 import PPO

        n_steps = max(64, int(n_steps))
        batch_size = min(256, n_steps)
        while n_steps % batch_size != 0 and batch_size > 1:
            batch_size //= 2
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=entropy_coef,
            policy_kwargs=ppo_policy_kwargs(hidden),
            seed=seed,
            device=device,
            verbose=0,
        )
    if name == "SAC":
        return build_sac(env, learning_rate, hidden, seed, device)
    if name == "TD3":
        return build_td3(env, learning_rate, hidden, seed, device)
    if name in {"LSTM", "RECURRENT_PPO"}:
        return build_recurrent_ppo(env, learning_rate, hidden, seed, device)
    raise KeyError(algorithm)
