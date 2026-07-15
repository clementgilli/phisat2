from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Prefixes that identify ViT-based architectures in TerraTorch.
# CNN-based backbones (seco, ssl4eos12, satlas…) are NOT listed here.
_VIT_PREFIXES = ("terramind", "dofa", "prithvi", "clay", "ssl4eos12_vit")


class TerraTorchBackboneEncoder(nn.Module):
    """
    Universal TerraTorch backbone wrapper — CNN and ViT teachers.

    Output format depends on `self.output_type`:

        "cnn" → list[Tensor]
            Multi-scale feature maps, identical format to PhiSatNetEncoder.
            Compatible with multi-scale MSE distillation for CNN→CNN KD.

        "vit" → dict {"cls_token": Tensor, "patch_tokens": Tensor | None}
            Raw tokens from the ViT encoder.
            Compatible with CrossArchKDModule for ViT→CNN KD.
    """

    def __init__(
        self,
        backbone: str,
        *,
        pretrained: bool,
        input_bands: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__()

        from phisat2.models.registry import get_model_bands

        try:
            from terratorch import BACKBONE_REGISTRY
        except ImportError as exc:
            raise ImportError("TerraTorch is required.  Run `make install`.") from exc

        build_kwargs: dict[str, Any] = dict(kwargs)

        # ── Band intersection ─────────────────────────────────────────────────
        reg_bands = get_model_bands(backbone)
        if not reg_bands:
            raise ValueError(
                f"Model '{backbone}' has no pretrain_bands defined in the registry."
            )

        _input_index = {band: i for i, band in enumerate(input_bands)}
        valid_bands: list[str] = []
        self.valid_indices: list[int] = []
        for band in reg_bands:
            if band in _input_index:
                valid_bands.append(band)
                self.valid_indices.append(_input_index[band])


        if not valid_bands:
            raise ValueError(
                f"No band overlap between input_bands={input_bands} "
                f"and model bands={list(reg_bands)}."
            )
            
        print(f"[TerraTorchBackboneEncoder] Using bands: {valid_bands} (indices: {self.valid_indices})")

        # ── Architecture type (drives output normalisation) ───────────────────
        # Determines which _normalize_* branch runs in forward().
        self.output_type: str = (
            "vit"
            if any(backbone.lower().startswith(p) for p in _VIT_PREFIXES)
            else "cnn"
        )

        # ── TerraTorch model-specific build kwargs ────────────────────────────
        self.is_multimodal = backbone.startswith("terramind")
        self.modality_name = "S2L1C"

        if self.is_multimodal:
            # TerraMind expects a modality→bands dict
            build_kwargs.setdefault("modalities", [self.modality_name])
            build_kwargs.setdefault("bands", {self.modality_name: valid_bands})

        elif backbone.startswith(("dofa_", "clay_", "ssl4eos12_vit_")):
            build_kwargs.setdefault("model_bands", valid_bands)

        elif "resnet" in backbone:
            # CNN-based self-supervised models (ResNet backbone)
            build_kwargs.setdefault("model_bands", valid_bands)
            build_kwargs.setdefault("out_indices", [1, 2, 3, 4])

        elif backbone.startswith("prithvi"):
            build_kwargs.setdefault("bands", valid_bands)

        else:
            raise ValueError(f"Unknown TerraTorch backbone '{backbone}'.")

        self.backbone = BACKBONE_REGISTRY.build(
            backbone, pretrained=pretrained, **build_kwargs
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> list[torch.Tensor] | dict[str, torch.Tensor | None]:
        x = x[:, self.valid_indices, :, :]

        raw = (
            self.backbone({self.modality_name: x})
            if self.is_multimodal
            else self.backbone(x)
        )

        return self._normalize_cnn(raw) if self.output_type == "cnn" else self._normalize_vit(raw)

    # ── Output normalisation ──────────────────────────────────────────────────

    def _normalize_cnn(self, raw: Any) -> list[torch.Tensor]:
        """
        CNN backbone output → list[Tensor] of spatial feature maps.

        TerraTorch CNN wrappers (seco, ssl4eos12, satlas) can return:
          - list[Tensor]              already fine
          - tuple[Tensor, ...]        flat tuple of spatial maps
          - tuple[tuple[Tensor,...]]  nested (TerraTorch wrapper quirk)
          - dict with a "features" key
          - single Tensor             only the last stage
        """
        if isinstance(raw, list):
            return raw

        if isinstance(raw, tuple):
            # Flat tuple of 4-D spatial tensors: (f1, f2, f3, f4)
            if raw and isinstance(raw[0], torch.Tensor) and raw[0].ndim == 4:
                return list(raw)
            # Nested from TerraTorch wrappers: ((f1, f2, f3, f4),)
            if len(raw) == 1 and isinstance(raw[0], (list, tuple)):
                return list(raw[0])

        if isinstance(raw, torch.Tensor):
            return [raw]

        if isinstance(raw, dict):
            for key in ("features", "out", "encoder_features", "feature_maps"):
                if key in raw:
                    return self._normalize_cnn(raw[key])

        raise RuntimeError(
            f"[TerraTorchBackboneEncoder] Cannot normalise CNN output: "
            f"type={type(raw)}.  Add a new case to _normalize_cnn()."
        )

    def _normalize_vit(self, raw: Any) -> dict[str, torch.Tensor | None]:

        # ── dict structuré (DINOv2-style) ────────────────────────────────────
        if isinstance(raw, dict):
            cls   = raw.get("cls_token") or raw.get("x_norm_clstoken") or raw.get("global")
            patch = raw.get("patch_tokens") or raw.get("x_norm_patchtokens") or raw.get("local")
            if cls is not None:
                return {"cls_token": cls, "patch_tokens": patch}

        # ── list / tuple ──────────────────────────────────────────────────────
        if isinstance(raw, (list, tuple)):

            if len(raw) == 1:
                return self._normalize_vit(raw[0])

            if all(isinstance(f, torch.Tensor) for f in raw):
                first = raw[0]

                # (B, D) + (B, N, D) → cls et patches déjà séparés
                if first.ndim == 2:
                    return {"cls_token": raw[0], "patch_tokens": raw[1]}

                if first.ndim == 3:
                    tokens = raw[-1]             # (B, N, D) — last transformer layer
                    cls    = tokens.mean(dim=1)  # (B, D)    — global embedding via mean pool
                    return {"cls_token": cls, "patch_tokens": tokens}

                if first.ndim == 4:
                    last = raw[-1]
                    cls  = F.adaptive_avg_pool2d(last, 1).flatten(1)
                    return {"cls_token": cls, "patch_tokens": None}

            return self._normalize_vit(raw[-1])

        # ── Tensor brut ───────────────────────────────────────────────────────
        if isinstance(raw, torch.Tensor):
            if raw.ndim == 3:
                return {"cls_token": raw[:, 0], "patch_tokens": raw[:, 1:]}
            if raw.ndim == 2:
                return {"cls_token": raw, "patch_tokens": None}

        detail = f"type={type(raw)}"
        if hasattr(raw, "__len__"):
            detail += f", len={len(raw)}"
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            e0 = raw[0]
            detail += f", raw[0]={type(e0)}"
            if isinstance(e0, torch.Tensor):
                detail += f" shape={e0.shape} ndim={e0.ndim}"
            elif isinstance(e0, (list, tuple)):
                detail += f" len={len(e0)}"
                if len(e0) > 0 and isinstance(e0[0], torch.Tensor):
                    detail += f" [0].shape={e0[0].shape}"
        raise RuntimeError(
            f"[TerraTorchBackboneEncoder] Cannot normalise ViT output: {detail}. "
            f"Add a case to _normalize_vit()."
        )