from __future__ import annotations

import torch
import torch.nn as nn


class GlobalPoolingHead(nn.Module):
    def __init__(self, in_channels: int, out_features: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(in_channels, out_features)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        return self.head(self.flatten(self.pool(x)))
