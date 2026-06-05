from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

class TerraTorchBackboneEncoder(nn.Module):
    def __init__(self, backbone: str, *, pretrained: bool, input_bands: list[str], **kwargs: Any) -> None:
        super().__init__()
        from phisat2.models.registry import get_model_bands
        
        try:
            from terratorch import BACKBONE_REGISTRY
        except ImportError as exc:
            raise ImportError("TerraTorch is required. Run `make install`.") from exc

        build_kwargs: dict[str, Any] = dict(kwargs)
        
        self.is_multimodal = False
        self.modality_name = "S2L1C"

        self.valid_indices = []
        valid_bands = []
        
        reg_bands = get_model_bands(backbone)
        if reg_bands:
            for i, band in enumerate(input_bands):
                if band in reg_bands:
                    self.valid_indices.append(i)
                    valid_bands.append(band)
        else:
            raise ValueError(f"Model '{backbone}' has no pretrain_bands defined in the register.")

        if backbone.startswith("terramind"):
            self.is_multimodal = True
            build_kwargs.setdefault("modalities", [self.modality_name])
            build_kwargs.setdefault("bands", {self.modality_name: valid_bands})
            
        elif backbone.startswith("dofa_"):
            build_kwargs.setdefault("model_bands", valid_bands)
            
        elif backbone.startswith(("seco_", "ssl4eos12_", "satlas_")):
            build_kwargs.setdefault("model_bands", valid_bands)
            
        elif backbone.startswith("prithvi"):
            build_kwargs.setdefault("bands", valid_bands)
            
        else:
            build_kwargs.setdefault("in_chans", len(valid_bands))

        self.backbone = BACKBONE_REGISTRY.build(backbone, pretrained=pretrained, **build_kwargs)

    def forward(self, x: torch.Tensor) -> Any:
        x = x[:, self.valid_indices, :, :]

        if self.is_multimodal:
            output = self.backbone({self.modality_name: x})
        else:
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