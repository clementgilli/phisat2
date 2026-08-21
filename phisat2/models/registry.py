from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from phisat2.models.composite import ComposedModel
from phisat2.models.encoders.phisatnet_encoder import PhiSatNetEncoder
from phisat2.models.decoders.phisatnet_decoder import PhiSatNetDecoder
from phisat2.models.heads import MultiScaleClassificationHead
from phisat2.models.encoders.terratorch_backbones import TerraTorchBackboneEncoder
from phisat2.tasks import TaskSpec
from phisat2.tasks.specs import resolve_task_spec, guess_task_from_dataset
from phisat2.utils.weights import load_encoder_weights, load_decoder_weights


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelEntry:
    """Static metadata for a registered model."""
    name:           str
    role:           str                     # "student" | "teacher"
    description:    str
    pretrain_bands: tuple[str, ...] | None = None


@dataclass
class ModelBundle:
    """
    Type-safe container returned by build_model().
    Callers access fields by name — no more positional tuple-unpacking.

    task                    | model | teacher | student | decoders
    ─────────────────────────┼───────┼─────────┼─────────┼─────────
    pretrain_reconstruction  |  ✓    |         |         |
    knowledge_distillation          |       |    ✓    |    ✓    |
    domain_adaptation        |       |    ✓    |    ✓    |
    eval_domain_gap          |       |    ✓    |    ✓    |    ✓
    downstream               |  ✓    |         |         |
    """
    task:     str
    model:    nn.Module     | None = None
    teacher:  nn.Module     | None = None
    student:  nn.Module     | None = None
    student_before: nn.Module | None = None
    decoders: nn.ModuleDict | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Band definitions
# ─────────────────────────────────────────────────────────────────────────────

_PHISATNET_BANDS: tuple[str, ...] = (
    "PAN", "BLUE", "GREEN", "RED",
    "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD",
)

_TERRAMIND_BANDS: tuple[str, ...] = (
    "COASTAL_AEROSOL", "BLUE", "GREEN", "RED",
    "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD",
    "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2",
)

_SATLAS_BANDS: tuple[str, ...] = (
    "RED", "GREEN", "BLUE",
    "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
    "NIR_BROAD", "SWIR_1", "SWIR_2"
)

_SSL4EO_BANDS: tuple[str, ...] = (
    "COASTAL_AEROSOL", "BLUE", "GREEN", "RED",
    "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD",
    "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2",
)

_DOFA_BANDS: tuple[str, ...] = (
    "COASTAL_AEROSOL", "BLUE", "GREEN", "RED",
    "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD",
    "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2",
)

_CLAY_BANDS: tuple[str, ...] = (
    "COASTAL_AEROSOL", "BLUE", "GREEN", "RED",
    "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD",
    "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2",
)

_PRITHVI_BANDS: tuple[str, ...] = (
    "BLUE", "GREEN", "RED",
    "NIR_NARROW", "SWIR_1", "SWIR_2",
)

_SECO_BANDS: tuple[str, ...] = (
    "BLUE", "GREEN", "RED",
)
    

# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY: dict[str, ModelEntry] = {
    # ── Students ──────────────────────────────────────────────────────────
    "phisatnet": ModelEntry(
        name="phisatnet",
        role="student",
        description="Lightweight CNN encoder for PhiSat-2 onboard deployment.",
        pretrain_bands=_PHISATNET_BANDS,
    ),
    # ── Teachers ──────────────────────────────────────────────────────────
    
    ## ── CNN based teachers  ──
    "satlas_resnet50_sentinel2_si_ms_satlas": ModelEntry(
        name="satlas_resnet50_sentinel2_si_ms_satlas",
        role="teacher",
        description="Satlas ResNet-50 via TerraTorch.",
        pretrain_bands=_SATLAS_BANDS,
    ),
    "ssl4eos12_resnet50_sentinel2_all_dino": ModelEntry(
        name="ssl4eos12_resnet50_sentinel2_all_dino",
        role="teacher",
        description="SSL4EO-S12 ResNet-50 trained with DINO via TerraTorch.",
        pretrain_bands=_SSL4EO_BANDS,
    ),
    "ssl4eos12_resnet50_sentinel2_all_decur": ModelEntry(
        name="ssl4eos12_resnet50_sentinel2_all_decur",
        role="teacher",
        description="SSL4EO-S12 ResNet-50 trained with DeCur via TerraTorch.",
        pretrain_bands=_SSL4EO_BANDS,
    ),
    "ssl4eos12_resnet50_sentinel2_all_softcon": ModelEntry(
        name="ssl4eos12_resnet50_sentinel2_all_softcon",
        role="teacher",
        description="SSL4EO-S12 ResNet-50 trained with SoftContrast via TerraTorch.",
        pretrain_bands=_SSL4EO_BANDS,
    ),
    "ssl4eos12_resnet18_sentinel2_all_moco": ModelEntry(
        name="ssl4eos12_resnet18_sentinel2_all_moco",
        role="teacher",
        description="SSL4EO-S12 ResNet-18 trained with MoCo via TerraTorch.",
        pretrain_bands=_SSL4EO_BANDS,
    ),
    "ssl4eos12_resnet50_sentinel2_all_moco": ModelEntry(
        name="ssl4eos12_resnet50_sentinel2_all_moco",
        role="teacher",
        description="SSL4EO-S12 ResNet-50 trained with MoCo via TerraTorch.",
        pretrain_bands=_SSL4EO_BANDS,
    ),
    "seco_resnet18_sentinel2_rgb_seco": ModelEntry(
        name="seco_resnet18_sentinel2_rgb_seco",
        role="teacher",
        description="SeCo ResNet-18 via TerraTorch.",
        pretrain_bands=_SECO_BANDS,
    ),
    "seco_resnet50_sentinel2_rgb_seco": ModelEntry(
        name="seco_resnet50_sentinel2_rgb_seco",
        role="teacher",
        description="SeCo ResNet-50 via TerraTorch.",
        pretrain_bands=_SECO_BANDS,
    ),
    
    ## ── ViT based teachers ──
    "terramind_v1_tiny": ModelEntry( # 6.0 M params
        name="terramind_v1_tiny",
        role="teacher",
        description="TerraMind ViT-Tiny via TerraTorch.",
        pretrain_bands=_TERRAMIND_BANDS,
    ),
    "terramind_v1_small": ModelEntry( # 22.6 M params
        name="terramind_v1_small",
        role="teacher",
        description="TerraMind ViT-Small via TerraTorch.",
        pretrain_bands=_TERRAMIND_BANDS,
    ),
    "terramind_v1_base": ModelEntry( # 87.5 M params
        name="terramind_v1_base",
        role="teacher",
        description="TerraMind ViT-Base via TerraTorch.",
        pretrain_bands=_TERRAMIND_BANDS,
    ),
    "terramind_v1_large": ModelEntry( # 305 M params
        name="terramind_v1_large",
        role="teacher",
        description="TerraMind ViT-Large via TerraTorch.",
        pretrain_bands=_TERRAMIND_BANDS,
    ),
    "satlas_swin_t_sentinel2_si_ms": ModelEntry(
        name="satlas_swin_t_sentinel2_si_ms",
        role="teacher",
        description="Satlas Swin-Tiny via TerraTorch.",
        pretrain_bands=_SATLAS_BANDS,
    ),
    "satlas_swin_b_sentinel2_si_ms": ModelEntry(
        name="satlas_swin_b_sentinel2_si_ms",
        role="teacher",
        description="Satlas Swin-Base via TerraTorch.",
        pretrain_bands=_SATLAS_BANDS,
    ),
    "ssl4eos12_vit_small_patch16_224_sentinel2_all_dino": ModelEntry( # 23 M params
        name="ssl4eos12_vit_small_patch16_224_sentinel2_all_dino",
        role="teacher",
        description="SSL4EO-S12 ViT-Small trained with DINO via TerraTorch.",
        pretrain_bands=_SSL4EO_BANDS,
    ),
    "ssl4eos12_vit_small_patch16_224_sentinel2_all_moco": ModelEntry( # 23 M params
        name="ssl4eos12_vit_small_patch16_224_sentinel2_all_moco",
        role="teacher",
        description="SSL4EO-S12 ViT-Small trained with MoCo via TerraTorch.",
        pretrain_bands=_SSL4EO_BANDS,
    ),
    "dofa_small_patch16_224": ModelEntry( # 34.8 M params
        name="dofa_small_patch16_224",
        role="teacher",
        description="DOFA small ViT backbone via TerraTorch.",
        pretrain_bands=_DOFA_BANDS,
    ),
    "dofa_base_patch16_224": ModelEntry( # 111 M params
        name="dofa_base_patch16_224",
        role="teacher",
        description="DOFA base ViT backbone via TerraTorch.",
        pretrain_bands=_DOFA_BANDS,
    ),
    "dofa_large_patch16_224": ModelEntry( # 337 M params
        name="dofa_large_patch16_224",
        role="teacher",
        description="DOFA large ViT backbone via TerraTorch.",
        pretrain_bands=_DOFA_BANDS,
    ),
    "clay_v1_base": ModelEntry(
        name="clay_v1_base",
        role="teacher",
        description="CLAY v1 base backbone via TerraTorch.",
        pretrain_bands=_CLAY_BANDS,
    ),
    "prithvi_eo_v1_100": ModelEntry( # 86.2 M params
        name="prithvi_eo_v1_100",
        role="teacher",
        description="Prithvi EO v1 100 backbone via TerraTorch.",
        pretrain_bands=_PRITHVI_BANDS,
    ),
    "prithvi_eo_v2_300": ModelEntry( # 303 M params
        name="prithvi_eo_v2_300",
        role="teacher",
        description="Prithvi EO v2 300 backbone via TerraTorch.",
        pretrain_bands=_PRITHVI_BANDS,
    ),
    "prithvi_eo_v2_600": ModelEntry( # 631 M params
        name="prithvi_eo_v2_600",
        role="teacher",
        description="Prithvi EO v2 600 backbone via TerraTorch.",
        pretrain_bands=_PRITHVI_BANDS,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public utilities
# ─────────────────────────────────────────────────────────────────────────────

def list_models() -> list[ModelEntry]:
    return [REGISTRY[name] for name in sorted(REGISTRY)]


def get_model_bands(name: str) -> tuple[str, ...] | None:
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'.")
    return REGISTRY[name].pretrain_bands


# ─────────────────────────────────────────────────────────────────────────────
# Private builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_encoder(name: str, pretrained: bool, input_bands: list[str], base_channels: int = 16) -> nn.Module:
    if name == "phisatnet":
        if isinstance(input_bands, list):
            input_bands = len(input_bands)
        return PhiSatNetEncoder(in_channels=input_bands, base_channels=base_channels)
    return TerraTorchBackboneEncoder(name, pretrained=pretrained, input_bands=input_bands)


def _build_decoder(spec: TaskSpec, feature_channels: tuple[int, ...]) -> nn.Module:
    if spec.task in {"segmentation", "pixel_regression", "pretrain_reconstruction"}:
        return PhiSatNetDecoder(feature_channels, spec.num_outputs)
    return MultiScaleClassificationHead(feature_channels, spec.num_outputs)



# ─────────────────────────────────────────────────────────────────────────────
# Main factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    name: str,
    spec: TaskSpec,
    *,
    pretrained: bool,
    input_bands: list[str],
    teacher_bands: list[str] | None = None,
    weights_path:  str | None       = None,
    teacher_ckpt:  str | None       = None,
    student_ckpt:  str | None       = None,
    decoders:      list[str] | None = None,
    base_channels: int = 16,
) -> ModelBundle:
    """
    Central factory for all model configurations.
    Always returns a ModelBundle — no raw tuple unpacking in callers.

    Args:
        name          : Model name from REGISTRY (e.g. "phisatnet", "terramind_v1_base").
        spec          : TaskSpec describing the task and dataset.
        pretrained    : Whether to load pretrained backbone weights (teachers only).
        input_bands   : Spectral bands the model will receive as input.
        weights_path  : Encoder .pth or full Lightning .ckpt (SSL, DA, downstream).
        teacher_ckpt  : Encoder .pth for the teacher (eval_domain_gap only).
        student_ckpt  : Encoder .pth for the student (eval_domain_gap only).
                        Falls back to teacher_ckpt if None.
        decoders      : List of "dataset_name=path/to/ckpt" strings (eval_domain_gap only).
    """
    
    if base_channels != 16:
        print(f"\n{'='*70}")
        print(f" WARNING: NON-STANDARD STUDENT CAPACITY SELECTED")
        print(f" PhiSatNet base_channels is set to {base_channels} (default: 16).")
        print(f" You are training a scaled-up student to debug the capacity gap.")
        print(f"{'='*70}\n")
    
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Expected one of: {', '.join(sorted(REGISTRY))}."
        )

    entry = REGISTRY[name]

    # ── Phase 1 — Pretrain SSL ────────────────────────────────────────────
    if spec.task == "pretrain_reconstruction":
        if entry.role != "student":
            raise ValueError(
                f"Pretraining is reserved for 'student' models, "
                f"got '{name}' (role={entry.role})."
            )
        encoder = _build_encoder(name, pretrained=pretrained, input_bands=input_bands)
        head    = _build_decoder(spec, tuple(encoder.out_channels))
        return ModelBundle(task=spec.task, model=ComposedModel(encoder, head))

    # ── Phase 2 — Knowledge Distillation ─────────────────────────────────
    elif spec.task == "knowledge_distillation":
        if entry.role != "teacher":
            raise ValueError(
                f"KD requires a 'teacher' model as --model, "
                f"got '{name}' (role={entry.role})."
            )
        
        if not teacher_bands:
            raise ValueError(
                "Knowledge Distillation requires --teacher-bands to be specified "
                "in the datamodule (e.g. s2_bands for Sentinel-2)."
            )
        
        teacher = _build_encoder(name,        pretrained=True,  input_bands=teacher_bands)
        student = _build_encoder("phisatnet", pretrained=False, input_bands=input_bands, base_channels=base_channels)
        
        return ModelBundle(task=spec.task, teacher=teacher, student=student)

    # ── Phase 4 — Domain Adaptation ───────────────────────────────────────
    elif spec.task == "domain_adaptation":
        if not weights_path:
            raise ValueError(
                "DA requires --weights pointing to the pretrained SIM encoder. "
                "Both teacher (frozen) and student are initialised from it."
            )
        teacher = _build_encoder("phisatnet", pretrained=False, input_bands=8)        
        student = _build_encoder("phisatnet", pretrained=False, input_bands=8)        
        load_encoder_weights(teacher, weights_path)        
        load_encoder_weights(student, weights_path)
        
        return ModelBundle(task=spec.task, teacher=teacher, student=student)

    # ── Domain Gap Evaluation ─────────────────────────────────────────────
    elif spec.task == "eval_domain_gap":
        if not teacher_ckpt:
            raise ValueError("eval_domain_gap requires --teacher-ckpt.")
        
        teacher = _build_encoder("phisatnet", pretrained=False, input_bands=8)
        student = _build_encoder("phisatnet", pretrained=False, input_bands=8)  
        load_encoder_weights(teacher, teacher_ckpt)        
        load_encoder_weights(student, student_ckpt)

        decoders_dict = nn.ModuleDict()
        for dec_arg in (decoders or []):
            dataset_name, ckpt_path = dec_arg.split("=", 1)
            inferred_task = guess_task_from_dataset(dataset_name)
            dec_spec      = resolve_task_spec(inferred_task, dataset=dataset_name)
            head          = _build_decoder(dec_spec, tuple(teacher.out_channels))
            load_decoder_weights(head, ckpt_path)
            decoders_dict[dataset_name] = head

        return ModelBundle(
            task=spec.task, teacher=teacher, student=student, decoders=decoders_dict
        )

    # ── Downstream ────────────────────────────────────────────────────────
    else:
        target_name = "phisatnet" if entry.role == "teacher" else name
        if entry.role == "teacher":
            print(f"[INFO] Downstream: using 'phisatnet' architecture (ignoring '{name}').")

        encoder = _build_encoder(target_name, pretrained=False, input_bands=input_bands, base_channels=base_channels)
        if weights_path:
            load_encoder_weights(encoder, weights_path)

        head = _build_decoder(spec, tuple(encoder.out_channels))
        return ModelBundle(task=spec.task, model=ComposedModel(encoder, head))