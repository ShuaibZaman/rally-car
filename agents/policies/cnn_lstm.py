"""CNN encoder plus an LSTM, for the memory comparison scaffold."""

from __future__ import annotations

import torch
from torch import nn

from agents.policies.cnn import DrivingCNN


class CnnLstm(nn.Module):
    def __init__(self, out_dim: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = DrivingCNN(out_dim=hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, images: torch.Tensor, state=None):
        # images: (batch, time, H, W, 3) or (batch, time, 3, H, W)
        batch, time = images.shape[:2]
        flat = images.reshape(batch * time, *images.shape[2:])
        encoded = self.encoder(flat).reshape(batch, time, -1)
        output, state = self.lstm(encoded, state)
        return self.head(output), state
