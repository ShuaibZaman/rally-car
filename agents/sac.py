"""SAC builder. Training runs only when explicitly requested."""

from __future__ import annotations


def build_sac(env, learning_rate: float, hidden: list[int], seed: int, device: str):
    from stable_baselines3 import SAC

    return SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        policy_kwargs={"net_arch": hidden},
        seed=seed,
        device=device,
        verbose=0,
    )


def build_recurrent_ppo(env, learning_rate: float, hidden: list[int], seed: int, device: str):
    from sb3_contrib import RecurrentPPO

    return RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=learning_rate,
        policy_kwargs={"net_arch": hidden, "lstm_hidden_size": int(hidden[-1])},
        seed=seed,
        device=device,
        verbose=0,
    )
