from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleClassificationHead(nn.Module):

    def __init__(
        self,
        feature_channels: tuple[int, ...],
        out_features: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        in_features = 2 * sum(feature_channels)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, out_features),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        
        pooled = []
        for f in features:
            avg = self.avg_pool(f).flatten(1)   # (B, Ci)
            mx  = self.max_pool(f).flatten(1)   # (B, Ci)
            pooled.append(torch.cat([avg, mx], dim=1))   # (B, 2·Ci)

        x = torch.cat(pooled, dim=1)    # (B, 2·ΣCi)
        return self.classifier(x)