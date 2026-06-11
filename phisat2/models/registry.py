from __future__ import annotations

from dataclasses import dataclass

import re
import torch
import torch.nn as nn

from phisat2.models.composite import ComposedModel
from phisat2.models.encoders.phisatnet_encoder import PhiSatNetEncoder
from phisat2.models.decoders.phisatnet_decoder import PhiSatNetDecoder
from phisat2.models.heads import GlobalPoolingHead
from phisat2.models.encoders.terratorch_backbones import TerraTorchBackboneEncoder
from phisat2.tasks import TaskSpec
from phisat2.tasks.specs import resolve_task_spec, guess_task_from_dataset
from phisat2.utils.weights import load_encoder_weights, load_decoder_weights


@dataclass(frozen=True)
class ModelEntry:
    name: str
    role: str
    description: str
    pretrain_bands: Optional[Tuple[str, ...]] = None


_PHISATNET_BANDS = ["PAN", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"]
_TERRAMIND_BANDS = ("COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2")


REGISTRY = {
    # --- Students ---
    "phisatnet": ModelEntry(
        "phisatnet", "student", "Local compact PhiSat-2 CNN encoder baseline.", pretrain_bands=_PHISATNET_BANDS
    ),
    
    # --- Teachers ---
    "terramind_v1_tiny": ModelEntry(
        "terramind_v1_tiny", "teacher", "TerraTorch TerraMind tiny.", pretrain_bands=_TERRAMIND_BANDS
    ),
    "terramind_v1_small": ModelEntry(
        "terramind_v1_small", "teacher", "TerraTorch TerraMind small.", pretrain_bands=_TERRAMIND_BANDS
    ),
    "terramind_v1_base": ModelEntry(
        "terramind_v1_base", "teacher", "TerraTorch TerraMind base.", pretrain_bands=_TERRAMIND_BANDS
    ),
    "terramind_v1_large": ModelEntry(
        "terramind_v1_large", "teacher", "TerraTorch TerraMind large.", pretrain_bands=_TERRAMIND_BANDS
    ),
}


def list_models() -> list[ModelEntry]:
    return [REGISTRY[name] for name in sorted(REGISTRY)]

def get_model_bands(name: str) -> tuple[str, ...] | None:
    return REGISTRY[name].pretrain_bands

def _build_encoder(name: str, pretrained: bool, input_bands: list[str]) -> nn.Module:
    if name == "phisatnet":
        return PhiSatNetEncoder(in_channels=len(input_bands))
    else:
        return TerraTorchBackboneEncoder(name, pretrained=pretrained, input_bands=input_bands)

def _build_decoder(spec: TaskSpec, feature_channels: tuple[int, ...]) -> nn.Module:
    if spec.task in {"segmentation", "pixel_regression", "pretrain_reconstruction"}:
        return PhiSatNetDecoder(feature_channels, spec.num_outputs)
    else:
        return GlobalPoolingHead(feature_channels[-1], spec.num_outputs)

def build_model(
    name: str, 
    spec: TaskSpec, 
    *, 
    pretrained: bool, 
    input_bands: list[str], 
    weights_path: str | None = None,
    **kwargs
) -> nn.Module | tuple[nn.Module, nn.Module] | tuple[nn.Module, nn.Module, nn.ModuleDict]:
    
    if name not in REGISTRY:
        valid = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Expected one of: {valid}.")
    
    entry = REGISTRY[name]

    # ---------------------------------------------------------
    # PRE-TRAINING SSL
    # ---------------------------------------------------------
    if spec.task == "pretrain_reconstruction":
        if entry.role != "student":
            raise ValueError("Forbidden: Pretraining is reserved for 'student' models.")
        
        encoder = _build_encoder(name, pretrained=pretrained, input_bands=input_bands)
        head = _build_decoder(spec, tuple(encoder.out_channels))
        return ComposedModel(encoder, head)

    # ---------------------------------------------------------
    # DISTILLATION (KD)
    # ---------------------------------------------------------
    elif spec.task == "distillation_kd":
        if entry.role != "teacher":
            raise ValueError("Forbidden: KD requires a 'teacher' model.")
        
        teacher = _build_encoder(name, pretrained=True, input_bands=input_bands)
        student = _build_encoder("phisatnet", pretrained=False, input_bands=input_bands)
        
        if weights_path:
            load_encoder_weights(student, weights_path)
            
        return teacher, student

    # ---------------------------------------------------------
    # DOMAIN ADAPTATION (DA) - Sim to Real
    # ---------------------------------------------------------
    elif spec.task == "domain_adaptation":
        teacher = _build_encoder("phisatnet", pretrained=False, input_bands=input_bands)
        student = _build_encoder("phisatnet", pretrained=False, input_bands=input_bands)
        
        if not weights_path:
            raise ValueError("DA requires weights for the teacher.")
            
        load_encoder_weights(teacher, weights_path)
        load_encoder_weights(student, weights_path)
            
        return teacher, student

    # ---------------------------------------------------------
    # DOMAIN GAP EVALUATION
    # ---------------------------------------------------------
    elif spec.task == "eval_domain_gap":
        teacher_ckpt = kwargs.get("teacher_ckpt")
        student_ckpt = kwargs.get("student_ckpt")
        raw_decoders = kwargs.get("decoders") or []

        teacher = _build_encoder("phisatnet", pretrained=False, input_bands=input_bands)
        student = _build_encoder("phisatnet", pretrained=False, input_bands=input_bands)
        
        if teacher_ckpt:
            load_encoder_weights(teacher, teacher_ckpt)
        if student_ckpt:
            load_encoder_weights(student, student_ckpt)
        else:
            load_encoder_weights(student, teacher_ckpt)
            
        decoders_dict = nn.ModuleDict()
        
        for dec_arg in raw_decoders:
            dataset_name, ckpt_path = dec_arg.split("=")
            
            inferred_task = guess_task_from_dataset(dataset_name)            
            dec_spec = resolve_task_spec(inferred_task, dataset=dataset_name)
            
            head = _build_decoder(dec_spec, tuple(teacher.out_channels))
            load_decoder_weights(head, ckpt_path)
            decoders_dict[dataset_name] = head
                
        return teacher, student, decoders_dict

    # ---------------------------------------------------------
    # DOWNSTREAM
    # ---------------------------------------------------------
    else:    
        target_model_name = name
        if entry.role == "teacher":
            print("Downstream mode: Using 'phisatnet' architecture.")
            target_model_name = "phisatnet"

        encoder = _build_encoder(target_model_name, pretrained=False, input_bands=input_bands)
        if weights_path:
            load_encoder_weights(encoder, weights_path)
            
        head = _build_decoder(spec, tuple(encoder.out_channels))
        
        return ComposedModel(encoder, head)



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