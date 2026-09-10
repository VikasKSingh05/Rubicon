"""Fallback baseline: plain CNN on concatenated HSI + LiDAR channels.

Meets the spec's "fallback baseline" requirement — a cheap, robust model for
constrained environments and a report baseline to compare the fusion model
against. No attention / state-space components.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class BaselineCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden: int = 32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x_hsi: torch.Tensor, x_dem: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_hsi, x_dem], dim=1)
        x = self.embed(x)                 # (B, hidden, 16, 16)
        x = self.blocks(x)                # (B, hidden, 16, 16)
        return self.head(x.mean(dim=(2, 3)))