from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phisat2.models.blocks import ConvBlock


class SharedUNetDecoder(nn.Module):
    """Single spatial decoder used by segmentation and pixel regression models."""

    def __init__(self, feature_channels: tuple[int, int, int, int], output_channels: int) -> None:
        super().__init__()
        c0, c1, c2, c3 = feature_channels
        self.blocks = nn.ModuleList(
            [
                ConvBlock(c3 + c2, c2),
                ConvBlock(c2 + c1, c1),
                ConvBlock(c1 + c0, c0),
            ]
        )
        self.final_conv = nn.Conv2d(c0, output_channels, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"SharedUNetDecoder expects 4 feature maps, got {len(features)}.")
        high, skip1, skip2, x = features
        for skip, block in zip([skip2, skip1, high], self.blocks, strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.final_conv(x)
