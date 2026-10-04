"""Residual MLP: y = f(x) + W x inside each block, then a linear head."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = torch.tanh(self.fc1(x))
        return torch.tanh(self.fc2(hidden) + residual)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, width: int, blocks: int, out_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.input(x))
        return self.head(self.blocks(hidden))
