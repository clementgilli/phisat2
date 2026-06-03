from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from phisat2.models.composite import SharedDecoderModel
from phisat2.models.encoders.myriad2_full import Myriad2FullUNet
from phisat2.models.encoders.phisat_geoaware import PhiSat2GeoAwareEncoder
from phisat2.models.encoders.terratorch_backbones import TerraTorchBackboneEncoder
from phisat2.tasks import TaskSpec


@dataclass(frozen=True)
class ModelEntry:
    name: str
    role: str
    description: str
    shared_decoder: bool
    pretrain_bands: Optional[Tuple[str, ...]] = None
    pretrain_mean: Optional[Tuple[float, ...]] = None
    pretrain_std: Optional[Tuple[float, ...]] = None


#_PHISAT2_MEAN = (49.7866, 49.0253, 48.4297, 49.2364, 51.1648, 55.4065, 57.3572, 56.7808)
#_PHISAT2_STD  = (7.2800, 6.5203, 6.9570, 9.0981, 8.3858, 7.9555, 8.3155, 8.3664)

_TERRAMIND_BANDS = ("COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2")
_TERRAMIND_MEAN = (2357.089, 2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849, 3040.560, 3306.481, 1473.847, 506.070, 2472.825, 1838.929)
_TERRAMIND_STD  = (1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131, 1679.732, 1727.26, 1024.687, 442.165, 1331.411, 1160.419)



REGISTRY = {
    # --- Students ---
    "phisat2_geoaware": ModelEntry(
        "phisat2_geoaware", "student", "Local compact PhiSat-2 CNN encoder baseline.", True
    ),
    "myriad2_full_unet": ModelEntry(
        "myriad2_full_unet", "student", "Full-structure Myriad2 U-Net exception.", False
    ),
    
    # --- Teachers ---
    "terramind_v1_tiny": ModelEntry(
        "terramind_v1_tiny", "teacher", "TerraTorch TerraMind tiny.", True,
        pretrain_bands=_TERRAMIND_BANDS, pretrain_mean=_TERRAMIND_MEAN, pretrain_std=_TERRAMIND_STD
    ),
    "terramind_v1_small": ModelEntry(
        "terramind_v1_small", "teacher", "TerraTorch TerraMind small.", True,
        pretrain_bands=_TERRAMIND_BANDS, pretrain_mean=_TERRAMIND_MEAN, pretrain_std=_TERRAMIND_STD
    ),
    "terramind_v1_base": ModelEntry(
        "terramind_v1_base", "teacher", "TerraTorch TerraMind base.", True,
        pretrain_bands=_TERRAMIND_BANDS, pretrain_mean=_TERRAMIND_MEAN, pretrain_std=_TERRAMIND_STD
    ),
    "terramind_v1_large": ModelEntry(
        "terramind_v1_large", "teacher", "TerraTorch TerraMind large .", True,
        pretrain_bands=_TERRAMIND_BANDS, pretrain_mean=_TERRAMIND_MEAN, pretrain_std=_TERRAMIND_STD
    ),
}


def list_models() -> list[ModelEntry]:
    return [REGISTRY[name] for name in sorted(REGISTRY)]

def get_model_stats(name: str) -> tuple[tuple[str, ...] | None, tuple[float, ...] | None, tuple[float, ...] | None]:
    if name not in REGISTRY:
        valid = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown model '{name}' for stats extraction. Expected one of: {valid}.")
    entry = REGISTRY[name]
    return entry.pretrain_bands, entry.pretrain_mean, entry.pretrain_std

def _build_encoder(name: str, pretrained: bool, input_bands: list[str]) -> nn.Module:
    if name == "phisat2_geoaware":
        return PhiSat2GeoAwareEncoder(in_channels=len(input_bands))
    else:
        return TerraTorchBackboneEncoder(name, pretrained=pretrained, input_bands=input_bands)

def build_model(name: str, spec: TaskSpec, *, pretrained: bool, input_bands: list[str], weights_path: str | None = None) -> nn.Module:
    if name not in REGISTRY:
        valid = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Expected one of: {valid}.")
    
    entry = REGISTRY[name]

    # ---------------------------------------------------------
    # PRE-TRAINING SSL
    # ---------------------------------------------------------
    if spec.task == "pretrain_reconstruction":
        if entry.role != "student":
            raise ValueError(f"Forbidden : You're trying to pretrain '{name}' which is a {entry.role}. Pretraining is reserved for 'student' models.")
        
        encoder = _build_encoder(name, pretrained=pretrained, input_bands=input_bands)
        return SharedDecoderModel(encoder, spec)

    # ---------------------------------------------------------
    # DISTILLATION (KD)
    # ---------------------------------------------------------
    elif spec.task == "distillation_kd":
        if entry.role != "teacher":
            raise ValueError(f"Forbidden : The distillation task (Teacher Benchmarking) requires a 'teacher' model, not a '{entry.role}'.")
        
        teacher = _build_encoder(name, pretrained=True, input_bands=input_bands)
        
        student = _build_encoder("phisat2_geoaware", pretrained=False, input_bands=input_bands)
        
        raise NotImplementedError("Not implemented yet.")

    # ---------------------------------------------------------
    # DOWNSTREAM (Linear Probing)
    # ---------------------------------------------------------
    else:    
        target_model_name = name
        if entry.role == "teacher":
            print(f"Downstream mode : Using 'phisat2_geoaware' architecture.")
            target_model_name = "phisat2_geoaware"

        if target_model_name == "myriad2_full_unet":
            model = Myriad2FullUNet(output_channels=spec.num_outputs)
        else:
            encoder = _build_encoder(target_model_name, pretrained=False, input_bands=input_bands)
            model = SharedDecoderModel(encoder, spec)

        if weights_path:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            
            cleaned_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    cleaned_dict[k.replace("model.", "", 1)] = v
                elif k.startswith("student."):
                    cleaned_dict[k.replace("student.", "", 1)] = v
                else:
                    cleaned_dict[k] = v
            
            missing, unexpected = model.load_state_dict(cleaned_dict, strict=False)
                        
        return model



""" TODO
     # --- PHISAT-2 / MYRIAD BASES ---
    "phisat2_geoaware": ModelEntry(
        "phisat2_geoaware", "Local compact PhiSat-2 CNN encoder baseline.", True, 
        pretrain_mean=_PHISAT2_MEAN, pretrain_std=_PHISAT2_STD
    ),
    "myriad2_full_unet": ModelEntry(
        "myriad2_full_unet", "Full-structure Myriad2 U-Net exception.", False,
        pretrain_mean=_PHISAT2_MEAN, pretrain_std=_PHISAT2_STD
    ),
    
    # --- PRITHVI ---
    "prithvi_eo_v1_100": ModelEntry(
        "prithvi_eo_v1_100", "TerraTorch Prithvi EO 100M backbone.", True, 
        pretrain_mean=_PRITHVI_MEAN, pretrain_std=_PRITHVI_STD
    ),
    "prithvi_eo_tiny": ModelEntry(
        "prithvi_eo_tiny", "TerraTorch Prithvi EO tiny backbone.", True, 
        pretrain_mean=_PRITHVI_MEAN, pretrain_std=_PRITHVI_STD
    ),
    "prithvi_eo_v2_tiny_tl": ModelEntry(
        "prithvi_eo_v2_tiny_tl", "TerraTorch Prithvi EO v2 tiny TL backbone.", True, 
        pretrain_mean=_PRITHVI_MEAN, pretrain_std=_PRITHVI_STD
    ),
    "prithvi_eo_v2_100_tl": ModelEntry(
        "prithvi_eo_v2_100_tl", "TerraTorch Prithvi EO v2 100M TL backbone.", True, 
        pretrain_mean=_PRITHVI_MEAN, pretrain_std=_PRITHVI_STD
    ),
    "prithvi_swin_B": ModelEntry(
        "prithvi_swin_B", "TerraTorch Prithvi Swin-B backbone.", True, 
        pretrain_mean=_PRITHVI_MEAN, pretrain_std=_PRITHVI_STD
    ),
    "prithvi_swin_L": ModelEntry(
        "prithvi_swin_L", "TerraTorch Prithvi Swin-L backbone.", True, 
        pretrain_mean=_PRITHVI_MEAN, pretrain_std=_PRITHVI_STD
    ),
    
    # --- DOFA ---
    "dofa_small_patch16_224": ModelEntry(
        "dofa_small_patch16_224", "TerraTorch DOFA small ViT backbone.", True,
        pretrain_mean=_DOFA_MEAN, pretrain_std=_DOFA_STD
    ),
    "dofa_base_patch16_224": ModelEntry(
        "dofa_base_patch16_224", "TerraTorch DOFA base ViT backbone.", True,
        pretrain_mean=_DOFA_MEAN, pretrain_std=_DOFA_STD
    ),
    "dofa_large_patch16_224": ModelEntry(
        "dofa_large_patch16_224", "TerraTorch DOFA large ViT backbone.", True,
        pretrain_mean=_DOFA_MEAN, pretrain_std=_DOFA_STD
    ),
    
    # --- SECO ---
    "seco_resnet18_sentinel2_rgb_seco": ModelEntry(
        "seco_resnet18_sentinel2_rgb_seco", "TerraTorch SeCo Sentinel-2 ResNet-18 backbone.", True,
        pretrain_mean=_SECO_RGB_MEAN, pretrain_std=_SECO_RGB_STD
    ),
    "seco_resnet50_sentinel2_rgb_seco": ModelEntry(
        "seco_resnet50_sentinel2_rgb_seco", "TerraTorch SeCo Sentinel-2 ResNet-50 backbone.", True,
        pretrain_mean=_SECO_RGB_MEAN, pretrain_std=_SECO_RGB_STD
    ),
    
    # --- SSL4EO ---
    "ssl4eos12_resnet18_sentinel2_all_moco": ModelEntry(
        "ssl4eos12_resnet18_sentinel2_all_moco", "TerraTorch SSL4EO-S12 Sentinel-2 ResNet-18 backbone.", True,
        pretrain_mean=_SSL4EO_MEAN, pretrain_std=_SSL4EO_STD
    ),
    "ssl4eos12_resnet50_sentinel2_all_moco": ModelEntry(
        "ssl4eos12_resnet50_sentinel2_all_moco", "TerraTorch SSL4EO-S12 Sentinel-2 ResNet-50 backbone.", True,
        pretrain_mean=_SSL4EO_MEAN, pretrain_std=_SSL4EO_STD
    ),
    "ssl4eos12_vit_small_patch16_224_sentinel2_all_moco": ModelEntry(
        "ssl4eos12_vit_small_patch16_224_sentinel2_all_moco", "TerraTorch SSL4EO-S12 Sentinel-2 ViT-small backbone.", True,
        pretrain_mean=_SSL4EO_MEAN, pretrain_std=_SSL4EO_STD
    ),
    
    # --- SATLAS ---
    "satlas_resnet50_sentinel2_si_ms_satlas": ModelEntry(
        "satlas_resnet50_sentinel2_si_ms_satlas", "TerraTorch Satlas Sentinel-2 ResNet-50 backbone.", True,
        pretrain_mean=_SATLAS_MEAN, pretrain_std=_SATLAS_STD
    ),
    "satlas_swin_t_sentinel2_si_ms": ModelEntry(
        "satlas_swin_t_sentinel2_si_ms", "TerraTorch Satlas Sentinel-2 Swin-T backbone.", True,
        pretrain_mean=_SATLAS_MEAN, pretrain_std=_SATLAS_STD
    )
    """