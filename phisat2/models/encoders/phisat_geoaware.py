from __future__ import annotations

import torch
import torch.nn as nn

from phisat2.models.blocks import ConvBlock


class PhiSat2GeoAwareEncoder(nn.Module):
    """Compact local PhiSat-2 CNN encoder baseline."""

    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 16,
        channel_multipliers: tuple[int, int, int, int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        channels = [base_channels * multiplier for multiplier in channel_multipliers]
        self.out_channels = channels
        self.encoders = nn.ModuleList(
            [
                ConvBlock(in_channels, channels[0]),
                ConvBlock(channels[0], channels[1]),
                ConvBlock(channels[1], channels[2]),
            ]
        )
        self.pools = nn.ModuleList([nn.MaxPool2d(2), nn.MaxPool2d(2), nn.MaxPool2d(2)])
        self.bottleneck = ConvBlock(channels[2], channels[3])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for encoder, pool in zip(self.encoders, self.pools, strict=True):
            x = encoder(x)
            features.append(x)
            x = pool(x)
        features.append(self.bottleneck(x))
        return features
