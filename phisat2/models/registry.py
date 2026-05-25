from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from phisat2.models.composite import SharedDecoderModel
from phisat2.models.encoders.myriad2_full import Myriad2FullUNet
from phisat2.models.encoders.phisat_geoaware import PhiSat2GeoAwareEncoder
from phisat2.models.encoders.terratorch_backbones import TerraTorchBackboneEncoder
from phisat2.tasks import TaskSpec


@dataclass(frozen=True)
class ModelEntry:
    name: str
    description: str
    shared_decoder: bool


REGISTRY = {
    "phisat2_geoaware": ModelEntry("phisat2_geoaware", "Local compact PhiSat-2 CNN encoder baseline.", True),
    "terramind_v1_tiny": ModelEntry("terramind_v1_tiny", "TerraTorch TerraMind tiny with 8-channel input.", True),
    "terramind_v1_base": ModelEntry("terramind_v1_base", "TerraTorch TerraMind base with 8-channel input.", True),
    "prithvi_eo_v1_100": ModelEntry("prithvi_eo_v1_100", "TerraTorch Prithvi EO 100M backbone.", True),
    "prithvi_eo_tiny": ModelEntry("prithvi_eo_tiny", "TerraTorch Prithvi EO tiny backbone.", True),
    "myriad2_full_unet": ModelEntry("myriad2_full_unet", "Full-structure Myriad2 U-Net exception.", False),
}


def list_models() -> list[ModelEntry]:
    return [REGISTRY[name] for name in sorted(REGISTRY)]


def build_model(name: str, spec: TaskSpec, *, pretrained: bool) -> nn.Module:
    if name not in REGISTRY:
        valid = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Expected one of: {valid}.")
    if name == "myriad2_full_unet":
        if spec.task not in {"segmentation", "pixel_regression"}:
            raise ValueError("myriad2_full_unet preserves a spatial U-Net and only supports spatial tasks.")
        return Myriad2FullUNet(output_channels=spec.num_outputs)
    if name == "phisat2_geoaware":
        return SharedDecoderModel(PhiSat2GeoAwareEncoder(), spec)
    encoder = TerraTorchBackboneEncoder(name, pretrained=pretrained)
    return SharedDecoderModel(encoder, spec)
