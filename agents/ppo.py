"""Clipped PPO objective.

training/train.py runs Stable-Baselines3 PPO. This function exists so loss-level
experiments can call the same objective without treating the reward as a loss.
"""

from __future__ import annotations

import torch


def ppo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> dict[str, torch.Tensor]:
    ratio = torch.exp(new_logp - old_logp)
    centered = advantages - advantages.mean()
    scale = centered.std(unbiased=False) + 1e-8
    adv = centered / scale
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = (returns - values).pow(2).mean()
    entropy_mean = entropy.mean()
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "total": total,
    }
