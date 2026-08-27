from __future__ import annotations

from typing import Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from phisat2.tasks import TaskSpec


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CORAL — Correlation Alignment (Sun & Saenko, ECCV 2016).

    Aligns 1st-order (mean) and 2nd-order (covariance) statistics of two
    feature distributions.  Does NOT use the within-batch pairing.

    source, target : (B, D) — GAP-pooled feature vectors.

    L_CORAL = ||C_s - C_t||_F² / (4 D²)
    """
    B, D = source.shape

    # Centre both distributions
    source_c = source - source.mean(dim=0, keepdim=True)   # (B, D)
    target_c = target - target.mean(dim=0, keepdim=True)

    # Unbiased sample covariance matrices (D, D)
    denom    = max(B - 1, 1)
    C_source = (source_c.T @ source_c) / denom
    C_target = (target_c.T @ target_c) / denom

    # Normalised Frobenius distance
    return (C_source - C_target).pow(2).sum() / (4.0 * D * D)


def mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    sigmas: list[float] | None = None,
) -> torch.Tensor:
    """
    Multi-Kernel Maximum Mean Discrepancy (MK-MMD).
    (Gretton et al., 2012; Long et al. DAN, ICML 2015)

    Estimates ||μ_s - μ_t||²_H summed over multiple RBF kernels, which
    implicitly matches all moments of the two distributions via the RKHS
    inner product.  Does NOT use the within-batch pairing.

    source, target : (B, D) — GAP-pooled feature vectors.
    sigmas         : RBF bandwidth list. Defaults to [0.5, 1, 2, 4, 8]
                     (covers a wide range of feature scales).

    L_MMD = Σ_σ [ E[k_σ(s,s')] - 2·E[k_σ(s,t)] + E[k_σ(t,t')] ]
    """
    if sigmas is None:
        sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]

    def _rbf(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
        """(B, D) × (B', D) → (B, B') RBF kernel matrix, mean-reduced to scalar."""
        sq_dist = torch.cdist(X, Y, p=2).pow(2)
        return torch.exp(-sq_dist / (2.0 * sigma ** 2)).mean()

    mmd = source.new_zeros(())
    for sigma in sigmas:
        mmd = mmd + _rbf(source, source, sigma) \
                  - 2.0 * _rbf(source, target, sigma) \
                  + _rbf(target, target, sigma)

    return mmd / len(sigmas)


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────

class DomainAdaptationModule(L.LightningModule):
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        spec: TaskSpec,
        *,
        lr: float,
        weight_decay: float,
        method: Literal["mse", "coral", "mmd"] = "mse",
        level_processing: Literal["L1C", "L0"] = "L1C",
        loss_weights: dict[str, float] | None = None,
        mmd_sigmas: list[float] | None = None,   # only used when method="mmd"
    ) -> None:
        super().__init__()
        self.student      = student_model
        self.teacher      = teacher_model
        self.spec         = spec
        self.lr           = lr
        self.weight_decay = weight_decay
        self.level_processing = level_processing
        self.method       = method
        self.mmd_sigmas   = mmd_sigmas   # None → use default in mmd_loss

        if method not in {"mse", "coral", "mmd"}:
            raise ValueError(f"Unknown method '{method}'. Choose from: mse, coral, mmd.")

        self.loss_weights = loss_weights or {
            "enc_0":      1.0,
            "enc_1":      1.0,
            "enc_2":      1.0,
            "bottleneck": 1.0,
        }

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.save_hyperparameters(ignore=["student_model", "teacher_model"])
        print(f"[DA] method={method}, level_processing={self.level_processing}" + (
            f" | sigmas={mmd_sigmas or '[0.5,1,2,4,8]'}" if method == "mmd" else ""
        ))

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        return self.student(image)

    def _level_loss(
        self,
        f_target: torch.Tensor,   # (B, C, H, W)
        f_source:  torch.Tensor,   # (B, C, H, W) — from frozen teacher
    ) -> torch.Tensor:
        """Compute the per-level loss according to the chosen method."""

        if self.method == "mse":
            # Point-to-point alignment (exploits co-registration)
            return F.mse_loss(f_target, f_source)

        # Distributional methods: GAP → (B, C)
        s_gap = F.adaptive_avg_pool2d(f_target, 1).flatten(1)  # student
        t_gap = F.adaptive_avg_pool2d(f_source,  1).flatten(1)  # teacher

        if self.method == "coral":
            return coral_loss(s_gap, t_gap)

        # method == "mmd"
        return mmd_loss(s_gap, t_gap, sigmas=self.mmd_sigmas)

    def _shared_step(
        self, batch: dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        if self.level_processing == "L0":
            img_target = batch["real_L0"]
        else:
            img_target = batch["real"]
        
        img_source  = batch["sentinel2"]

        with torch.no_grad():
            feat_source = self.teacher(img_source)

        feat_target = self.student(img_target)

        losses = []
        for i, (layer_name, weight) in enumerate(self.loss_weights.items()):
            if i >= len(feat_source) or i >= len(feat_target):
                break

            lvl_loss = self._level_loss(feat_target[i], feat_source[i])
            losses.append(lvl_loss * weight)

            self.log(
                f"{prefix}_{layer_name}_loss", lvl_loss,
                on_step=False, on_epoch=True, sync_dist=True,
            )

        if not losses:
            raise RuntimeError(
                f"No features mapped. "
                f"loss_weights={list(self.loss_weights)}, "
                f"len(feat_source)={len(feat_source)}"
            )

        total = sum(losses)
        self.log(
            f"{prefix}_loss", total,
            prog_bar=True, on_step=False, on_epoch=True,
            sync_dist=True, batch_size=img_target.shape[0],
        )
        return total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}