from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleClassificationHead(nn.Module):

    def __init__(
        self,
        feature_channels: tuple[int, ...],
        out_features: int,
        dropout: float = 0.3,
        multi_level: bool = True,
    ) -> None:
        super().__init__()
        self.multi_level = multi_level
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if self.multi_level:
            in_features = 2 * sum(feature_channels)
        else:
            in_features = 2 * feature_channels[-1]

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, out_features),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        
        pooled = []
        features_to_process = features if self.multi_level else [features[-1]]
        
        for f in features_to_process:
            avg = self.avg_pool(f).flatten(1)   # (B, Ci)
            mx  = self.max_pool(f).flatten(1)   # (B, Ci)
            pooled.append(torch.cat([avg, mx], dim=1))   # (B, 2·Ci)

        x = torch.cat(pooled, dim=1)
        return self.classifier(x)