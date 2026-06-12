from __future__ import annotations

import torch
import torch.nn as nn

class GlobalPoolingHead(nn.Module):
    def __init__(self, in_channels: int, out_features: int) -> None:
        super().__init__()
        
        self.pool = nn.AdaptiveMaxPool2d(1)
        
        self.bn = nn.BatchNorm1d(in_channels)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.head = nn.Linear(in_channels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.bn(x)
        
        x = self.dropout(x)
        
        return self.head(x)