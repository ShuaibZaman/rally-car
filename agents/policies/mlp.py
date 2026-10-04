"""Trunk parameter counts used to keep architecture comparisons honest.

These counts are for a single MLP from observation to action. Stable-Baselines3
also builds a critic, so a saved bundle reports both numbers.
"""

from __future__ import annotations

import torch
from torch import nn


def trunk_parameter_count(in_dim: int, hidden: list[int], out_dim: int) -> int:
    sizes = [in_dim, *hidden, out_dim]
    total = 0
    for left, right in zip(sizes, sizes[1:]):
        total += left * right + right
    return int(total)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.Tanh())
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
