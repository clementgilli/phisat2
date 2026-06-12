from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from phisat2.models.composite import ComposedModel
from phisat2.models.encoders.phisatnet_encoder import PhiSatNetEncoder
from phisat2.models.decoders.phisatnet_decoder import PhiSatNetDecoder
from phisat2.models.heads import GlobalPoolingHead
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
    distillation_kd          |       |    ✓    |    ✓    |
    domain_adaptation        |       |    ✓    |    ✓    |
    eval_domain_gap          |       |    ✓    |    ✓    |    ✓
    downstream               |  ✓    |         |         |
    """
    task:     str
    model:    nn.Module     | None = None
    teacher:  nn.Module     | None = None
    student:  nn.Module     | None = None
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
    "terramind_v1_tiny": ModelEntry(
        name="terramind_v1_tiny",
        role="teacher",
        description="TerraMind ViT-Tiny via TerraTorch.",
        pretrain_bands=_TERRAMIND_BANDS,
    ),
    "terramind_v1_small": ModelEntry(
        name="terramind_v1_small",
        role="teacher",
        description="TerraMind ViT-Small via TerraTorch.",
        pretrain_bands=_TERRAMIND_BANDS,
    ),
    "terramind_v1_base": ModelEntry(
        name="terramind_v1_base",
        role="teacher",
        description="TerraMind ViT-Base via TerraTorch.",
        pretrain_bands=_TERRAMIND_BANDS,
    ),
    "terramind_v1_large": ModelEntry(
        name="terramind_v1_large",
        role="teacher",
        description="TerraMind ViT-Large via TerraTorch.",
        pretrain_bands=_TERRAMIND_BANDS,
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

def _build_encoder(name: str, pretrained: bool, input_bands: list[str]) -> nn.Module:
    if name == "phisatnet":
        return PhiSatNetEncoder(in_channels=len(input_bands))
    return TerraTorchBackboneEncoder(name, pretrained=pretrained, input_bands=input_bands)


def _build_decoder(spec: TaskSpec, feature_channels: tuple[int, ...]) -> nn.Module:
    if spec.task in {"segmentation", "pixel_regression", "pretrain_reconstruction"}:
        return PhiSatNetDecoder(feature_channels, spec.num_outputs)
    return GlobalPoolingHead(feature_channels[-1], spec.num_outputs)


# ─────────────────────────────────────────────────────────────────────────────
# Main factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    name: str,
    spec: TaskSpec,
    *,
    pretrained: bool,
    input_bands: list[str],
    weights_path:  str | None       = None,
    teacher_ckpt:  str | None       = None,
    student_ckpt:  str | None       = None,
    decoders:      list[str] | None = None,
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
    elif spec.task == "distillation_kd":
        if entry.role != "teacher":
            raise ValueError(
                f"KD requires a 'teacher' model as --model, "
                f"got '{name}' (role={entry.role})."
            )
        if not weights_path:
            raise ValueError(
                "KD requires --weights pointing to the Phase 1 SSL encoder checkpoint."
            )
        teacher = _build_encoder(name,        pretrained=True,  input_bands=input_bands)
        student = _build_encoder("phisatnet", pretrained=False, input_bands=input_bands)
        load_encoder_weights(student, weights_path)
        return ModelBundle(task=spec.task, teacher=teacher, student=student)

    # ── Phase 4 — Domain Adaptation ───────────────────────────────────────
    elif spec.task == "domain_adaptation":
        if not weights_path:
            raise ValueError(
                "DA requires --weights pointing to the pretrained SIM encoder. "
                "Both teacher (frozen) and student are initialised from it."
            )
        teacher = _build_encoder("phisatnet", pretrained=False, input_bands=input_bands)
        student = _build_encoder("phisatnet", pretrained=False, input_bands=input_bands)
        load_encoder_weights(teacher, weights_path)
        load_encoder_weights(student, weights_path)
        
        return ModelBundle(task=spec.task, teacher=teacher, student=student)

    # ── Domain Gap Evaluation ─────────────────────────────────────────────
    elif spec.task == "eval_domain_gap":
        if not teacher_ckpt:
            raise ValueError("eval_domain_gap requires --teacher-ckpt.")

        teacher = _build_encoder("phisatnet", pretrained=False, input_bands=input_bands)
        student = _build_encoder("phisatnet", pretrained=False, input_bands=input_bands)
        load_encoder_weights(teacher, teacher_ckpt)
        load_encoder_weights(student, student_ckpt or teacher_ckpt)

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

        encoder = _build_encoder(target_name, pretrained=False, input_bands=input_bands)
        if weights_path:
            load_encoder_weights(encoder, weights_path)

        head = _build_decoder(spec, tuple(encoder.out_channels))
        return ModelBundle(task=spec.task, model=ComposedModel(encoder, head))



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