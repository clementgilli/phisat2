from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn


class FeaturePyramidAdapter(nn.Module):
    """Normalize arbitrary encoder outputs into one shared decoder contract."""

    def __init__(self, channels: tuple[int, int, int, int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.channels = channels
        self.projections = nn.ModuleList([nn.LazyConv2d(channel, kernel_size=1) for channel in channels])

    def forward(self, features: Any, image_size: tuple[int, int]) -> list[torch.Tensor]:
        normalized = self._select_four(features)
        pyramid = []
        for index, feature in enumerate(normalized):
            feature = self._to_spatial(feature)
            target_size = (max(1, image_size[0] // (2**index)), max(1, image_size[1] // (2**index)))
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            pyramid.append(self.projections[index](feature))
        return pyramid

    def _select_four(self, features: Any) -> list[torch.Tensor]:
        if torch.is_tensor(features):
            return [features, features, features, features]
        if isinstance(features, dict):
            values = list(features.values())
        elif isinstance(features, Sequence):
            values = list(features)
        else:
            raise TypeError(f"Unsupported encoder feature type: {type(features)!r}")
        if not values:
            raise ValueError("Encoder returned no features.")
        if len(values) >= 4:
            return list(values[-4:])
        return [values[0]] * (4 - len(values)) + values

    @staticmethod
    def _to_spatial(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 4:
            return feature
        if feature.ndim != 3:
            raise ValueError(f"Expected 3D token or 4D spatial feature tensor, got shape {tuple(feature.shape)}")
        batch, tokens, channels = feature.shape
        side = int(math.sqrt(tokens))
        if side * side != tokens:
            raise ValueError(f"Token count {tokens} cannot be reshaped to a square feature map.")
        return feature.transpose(1, 2).reshape(batch, channels, side, side)
