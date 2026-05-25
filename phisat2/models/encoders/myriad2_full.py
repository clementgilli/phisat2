from __future__ import annotations

import torch
import torch.nn as nn

from phisat2.models.blocks import ConvBlock


class Myriad2FullUNet(nn.Module):
    """Full-structure U-Net exception matching unet_myriad2_baseline topology."""

    def __init__(self, in_channels: int = 8, output_channels: int = 4) -> None:
        super().__init__()
        channels = [16, 32, 32, 48]
        self.encoders = nn.ModuleList(
            [
                ConvBlock(in_channels, channels[0]),
                ConvBlock(channels[0], channels[1]),
                ConvBlock(channels[1], channels[2]),
            ]
        )
        self.pools = nn.ModuleList([nn.MaxPool2d(2), nn.MaxPool2d(2), nn.MaxPool2d(2)])
        self.bottleneck = ConvBlock(channels[2], channels[3])
        self.upsamplers = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[3], channels[3], kernel_size=2, stride=2),
                nn.ConvTranspose2d(channels[2], channels[2], kernel_size=2, stride=2),
                nn.ConvTranspose2d(channels[1], channels[1], kernel_size=2, stride=2),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                ConvBlock(channels[3] + channels[2], channels[2]),
                ConvBlock(channels[2] + channels[1], channels[1]),
                ConvBlock(channels[1] + channels[0], channels[0]),
            ]
        )
        self.final_conv = nn.Conv2d(channels[0], output_channels, kernel_size=1)
        self.classifier = nn.Conv2d(channels[0], output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder, pool in zip(self.encoders, self.pools, strict=True):
            x = encoder(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for upsampler, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips), strict=True):
            x = upsampler(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.final_conv(x)
