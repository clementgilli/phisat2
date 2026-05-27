from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

SENTINEL2_EIGHT_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A")
DOFA_EIGHT_BANDS = ("BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW")


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
        if backbone.startswith("dofa_"):
            build_kwargs.setdefault("model_bands", list(DOFA_EIGHT_BANDS[:in_channels]))
        else:
            build_kwargs.setdefault("in_chans", in_channels)
        if backbone.startswith(("seco_", "ssl4eos12_", "satlas_")):
            build_kwargs.setdefault("model_bands", list(SENTINEL2_EIGHT_BANDS[:in_channels]))
        if backbone.startswith("prithvi"):
            build_kwargs.setdefault("bands", list(range(in_channels)))
        if backbone.startswith("terramind"):
            build_kwargs.setdefault("modalities", [])
        self.backbone = BACKBONE_REGISTRY.build(backbone, pretrained=pretrained, **build_kwargs)
        self.out_channels = None

    def forward(self, x: torch.Tensor) -> Any:
        output = self.backbone(x)
        if isinstance(output, dict):
            for key in ("features", "out", "encoder_features"):
                if key in output:
                    output = output[key]
                    break
        prepare = getattr(self.backbone, "prepare_features_for_image_model", None)
        if prepare is not None:
            output = prepare(output)
        return output
