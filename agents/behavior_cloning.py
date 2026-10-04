"""Behavior cloning from the pure-pursuit expert.

Losses: MSE, MAE, Huber. Optimizers: Adam, AdamW, SGD.
A full imitation run is not started by the scaffold tests.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from agents.classical import PurePursuit
from agents.policies.mlp import MLP

LOSS_NAMES = ("mse", "mae", "huber")
OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}


def regression_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        return nn.functional.mse_loss(pred, target)
    if kind == "mae":
        return nn.functional.l1_loss(pred, target)
    if kind == "huber":
        return nn.functional.smooth_l1_loss(pred, target)
    raise KeyError(kind)


def collect_demonstrations(env, steps: int, expert: PurePursuit | None = None) -> tuple[np.ndarray, np.ndarray]:
    expert = expert or PurePursuit()
    obs, _info = env.reset()
    observations = []
    actions = []
    for _ in range(steps):
        action = expert.act(env)
        observations.append(np.array(obs, dtype=np.float32))
        actions.append(np.array(action, dtype=np.float32))
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            obs, _info = env.reset()
    return np.stack(observations), np.stack(actions)


def train_bc_step(model: MLP, observations: np.ndarray, actions: np.ndarray, optimizer_name: str, loss_name: str, lr: float = 1e-3) -> float:
    optimizer_cls = OPTIMIZERS[optimizer_name]
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    pred = model(torch.as_tensor(observations))
    target = torch.as_tensor(actions)
    loss = regression_loss(pred, target, loss_name)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def copy_matching_parameters(source: nn.Module, target: nn.Module) -> int:
    """Copy tensors that share a name and shape. Returns how many were copied."""
    source_params = dict(source.named_parameters())
    copied = 0
    for name, param in target.named_parameters():
        match = source_params.get(name)
        if match is not None and match.shape == param.shape:
            param.data.copy_(match.data)
            copied += 1
    return copied
