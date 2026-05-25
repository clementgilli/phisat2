from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TerraTorchBackboneEncoder(nn.Module):
    """Lazy wrapper around TerraTorch backbones.

    Local files under pretrain/weights are intentionally not accepted here. Use
    TerraTorch's registry and remote/pretrained behavior only.
    """

    def __init__(self, backbone: str, *, pretrained: bool, in_channels: int = 8, **kwargs: Any) -> None:
        super().__init__()
        try:
            from terratorch import BACKBONE_REGISTRY
        except ImportError as exc:
            raise ImportError("TerraTorch is required for TerraTorch-backed models. Run `make install`.") from exc

        build_kwargs: dict[str, Any] = dict(kwargs)
        if backbone.startswith("terramind"):
            build_kwargs.setdefault("modalities", [])
            build_kwargs.setdefault("in_chans", in_channels)
        self.backbone = BACKBONE_REGISTRY.build(backbone, pretrained=pretrained, **build_kwargs)
        self.out_channels = None

    def forward(self, x: torch.Tensor) -> Any:
        output = self.backbone(x)
        if isinstance(output, dict):
            for key in ("features", "out", "encoder_features"):
                if key in output:
                    return output[key]
        return output
