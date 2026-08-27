from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phisat2.models.blocks import ConvBlock

class PhiSatNetDecoder(nn.Module):

    def __init__(self, feature_channels: tuple[int, int, int, int], output_channels: int) -> None:
        super().__init__()
        c0, c1, c2, c3 = feature_channels
        
        self.upsamplers = nn.ModuleList(
            [
                nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2),
                nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2),
                nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
            ]
        )
        
        self.decoders = nn.ModuleList(
            [
                ConvBlock(c3 + c2, c2),
                ConvBlock(c2 + c1, c1),
                ConvBlock(c1 + c0, c0),
            ]
        )
        
        self.final_conv = nn.Conv2d(c0, output_channels, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(f"PhiSatNetDecoder expects 4 feature maps, got {len(features)}.")
            
        skip0, skip1, skip2, x = features
        
        reversed_skips = [skip2, skip1, skip0]
        
        for upsampler, decoder, skip in zip(self.upsamplers, self.decoders, reversed_skips, strict=True):
            
            x = upsampler(x)    
            x = decoder(torch.cat([x, skip], dim=1))
            
        return self.final_conv(x)