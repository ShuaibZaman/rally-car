"""Small CNN for an 84x84x3 top-down image. Scaffold for the vision path."""

from __future__ import annotations

import torch
from torch import nn


class DrivingCNN(nn.Module):
    def __init__(self, out_dim: int = 3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            probe = self.conv(torch.zeros(1, 3, 84, 84))
            flat = int(probe.shape[1])
        self.head = nn.Sequential(nn.Linear(flat, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        image = image.float() / 255.0
        return self.head(self.conv(image))
