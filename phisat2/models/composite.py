from __future__ import annotations

import torch
import torch.nn as nn

from phisat2.models.adapters.feature_pyramid import FeaturePyramidAdapter
from phisat2.models.decoders.shared_unet import SharedUNetDecoder
from phisat2.models.heads import GlobalPoolingHead
from phisat2.tasks import TaskSpec


class SharedDecoderModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        spec: TaskSpec,
        *,
        feature_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = FeaturePyramidAdapter(feature_channels)
        if spec.task in {"segmentation", "pixel_regression"}:
            self.head = SharedUNetDecoder(feature_channels, spec.num_outputs)
        else:
            self.head = GlobalPoolingHead(feature_channels[-1], spec.num_outputs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        pyramid = self.adapter(features, image.shape[-2:])
        return self.head(pyramid)
