from __future__ import annotations

import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, layer_scale_init_value: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.BatchNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, 4 * channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma.view(1, -1, 1, 1) * x
        return identity + x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.channel_proj: nn.Module
        if in_channels != out_channels:
            self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_proj = nn.Identity()
        self.convnext_block = ConvNeXtBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnext_block(self.channel_proj(x))
