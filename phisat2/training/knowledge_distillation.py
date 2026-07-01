from __future__ import annotations

from typing import Any, Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def _attention_map(feat: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) → L2-normalised attention map (B, H, W)."""
    attn = feat.pow(2).mean(dim=1)
    return F.normalize(attn.flatten(1), p=2, dim=1).view_as(attn)


def _at_loss(
    s_feats: list[torch.Tensor],
    t_feats: list[torch.Tensor],
    weights: list[float],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Attention Transfer (Zagoruyko & Komodakis, 2016).
    MSE between L2-normalised spatial attention maps.
    Student pooled to teacher resolution — no upscaling, no projector.
    """
    per_level: list[torch.Tensor] = []
    total = s_feats[0].new_zeros(())

    for s_f, t_f, w in zip(s_feats, t_feats, weights):
        s_attn = _attention_map(s_f)
        t_attn = _attention_map(t_f)
        if s_attn.shape[-2:] != t_attn.shape[-2:]:
            s_attn = F.adaptive_avg_pool2d(
                s_attn.unsqueeze(1), t_attn.shape[-2:]
            ).squeeze(1)
        lvl = F.mse_loss(s_attn, t_attn.detach())
        per_level.append(lvl)
        total = total + w * lvl

    return total / sum(weights), per_level


def _rkd_loss(
    s_feat: torch.Tensor,
    t_feat: torch.Tensor,
) -> torch.Tensor:
    """
    Relational KD — distance-wise (Park et al., CVPR 2019).
    Pairwise distance structure on the deepest level only.
    No projector, no channel alignment — robust to large capacity gaps.
    """
    s = F.adaptive_avg_pool2d(s_feat, 1).flatten(1)
    t = F.adaptive_avg_pool2d(t_feat, 1).flatten(1)
    s_d = torch.cdist(s, s, p=2)
    t_d = torch.cdist(t, t, p=2)
    s_d = s_d / (s_d.mean() + 1e-6)
    t_d = t_d / (t_d.mean() + 1e-6)
    return F.smooth_l1_loss(s_d, t_d.detach())


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck (low-rank) projector
# ─────────────────────────────────────────────────────────────────────────────

class BottleneckProjector(nn.Module):
    """
    Low-rank projector: Conv1×1(C_in→r) → BN → GELU → Conv1×1(r→C_out) → BN.

    Why this exists
    ----------------
    A direct C_in×C_out projector explodes whenever one side has a large
    channel count. For PhiSatNet (128ch, last level) ↔ ResNet-50 (2048ch),
    a direct 1×1 conv costs 128*2048 = 262,144 params — more than the
    ENTIRE 199K student backbone, for a single projector at a single level.

    Factorising through a rank-r bottleneck costs C_in*r + r*C_out instead
    of C_in*C_out. With r=32 the same level drops from 262K to ~70K params.

    Side benefit: a low-rank bottleneck has less capacity to "absorb" a
    lazy/weak backbone's features, since everything must compress through
    a narrow r-dimensional space — a mild regulariser against the
    lazy-projector failure mode discussed earlier.
    """

    def __init__(self, c_in: int, c_out: int, rank: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(c_in, rank, kernel_size=1, bias=False),
            nn.BatchNorm2d(rank),
            nn.GELU(),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(rank, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(self.reduce(x))


def _default_rank(c_in: int, c_out: int, min_rank: int = 16, divisor: int = 4) -> int:
    """
    Heuristic bottleneck rank: ~1/4 of the smaller channel count, floored
    at min_rank, capped at min(c_in, c_out) so the bottleneck never widens
    the layer instead of narrowing it.
    """
    r = max(min_rank, min(c_in, c_out) // divisor)
    return min(r, c_in, c_out)


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────

class KDModule(L.LightningModule):
    """
    CNN→CNN multi-scale feature distillation.

    Three composable losses (toggle with use_* flags):
        Cosine — alignment via Conv1×1 projector + cosine distance.
        AT     — attention map transfer, no projector.
        RKD    — pairwise distance structure, no projector.

    Projector direction
    -------------------
    Both directions compare at TEACHER spatial resolution (smaller, no upscaling).

    "student_to_teacher"
        student (C_s, H_s) → AvgPool → (C_s, H_t) → Conv1×1 → (C_t, H_t)
        compare with teacher (C_t, H_t)
        Projector LR: 10× (random init → fast catch-up).
        Risk: projector compensates a lazy student backbone.

    "teacher_to_student"
        teacher (C_t, H_t) → Conv1×1 → (C_s, H_t)  [target]
        student (C_s, H_s) → AvgPool → (C_s, H_t)   [prediction]
        Projector warms up for `projector_warmup_epochs` (student frozen),
        then is frozen for the rest of training.
        This creates a fixed, meaningful target in student-channel space
        without the lazy-projector problem.

    Auto-lambda
    -----------
    Each active loss is divided by its value at the first training step
    → all terms start at 1.0, no manual λ tuning.
    Scales saved as buffers and restored from checkpoints.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        spec: Any,
        *,
        proj_dir: Literal["student_to_teacher", "teacher_to_student"] = "student_to_teacher",
        use_cosine: bool = True,
        use_at:     bool = False,
        use_rkd:    bool = False,
        lr:                      float       = 1e-4,
        weight_decay:            float       = 1e-4,
        level_weights:           list[float] | None = None,
        warmup_epochs:           int         = 5,
        projector_warmup_epochs: int         = 1,   # teacher_to_student only
        projector_rank:          int | None  = None,  # None = auto heuristic per level
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["student_model", "teacher_model"])

        if not (use_cosine or use_at or use_rkd):
            raise ValueError("Enable at least one loss (use_cosine / use_at / use_rkd).")

        self.student = student_model
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # ── Infer feature shapes via dummy pass ───────────────────────────────
        # S2 images are resized to 10m resolution before teacher forward:
        # at 10m/px the same footprint as 224px@4.75m → 224*4.75/10 ≈ 106px
        _DUMMY_S = 224
        _DUMMY_T = int(_DUMMY_S * 0.475)   # ~106px

        with torch.no_grad():
            s_feats = self.student(torch.zeros(1, 8,  _DUMMY_S, _DUMMY_S))
            t_feats = self.teacher(torch.zeros(1, 13, _DUMMY_T, _DUMMY_T))

        if len(s_feats) != len(t_feats):
            raise ValueError(
                f"Level mismatch: student={len(s_feats)}, teacher={len(t_feats)}.\n"
                f"  student: {[tuple(f.shape) for f in s_feats]}\n"
                f"  teacher: {[tuple(f.shape) for f in t_feats]}"
            )

        self.n_levels      = len(s_feats)
        self.level_weights = level_weights or [1.0] * self.n_levels

        # ── Projectors ────────────────────────────────────────────────────────
        # Both directions compare at TEACHER spatial resolution.
        # student_to_teacher : project student channels → teacher channels
        # teacher_to_student : project teacher channels → student channels
        self.projectors = nn.ModuleList()
        print(f"\n[KDModule] proj_dir={proj_dir}  "
              f"losses: cosine={use_cosine}  at={use_at}  rkd={use_rkd}")

        total_direct     = 0
        total_bottleneck = 0

        for lvl, (s_f, t_f) in enumerate(zip(s_feats, t_feats)):
            C_s, H_s = s_f.shape[1], s_f.shape[2]
            C_t, H_t = t_f.shape[1], t_f.shape[2]

            c_in, c_out = (C_s, C_t) if proj_dir == "student_to_teacher" else (C_t, C_s)
            rank = projector_rank or _default_rank(c_in, c_out)

            proj = BottleneckProjector(c_in, c_out, rank)
            self.projectors.append(proj)

            direct_p = c_in * c_out
            bneck_p  = c_in * rank + rank * c_out
            total_direct     += direct_p
            total_bottleneck += bneck_p

            if proj_dir == "student_to_teacher":
                desc = (f"s({C_s}ch,{H_s}px) → pool → ({C_s}ch,{H_t}px)"
                        f" → bottleneck(r={rank}) → ({C_t}ch,{H_t}px) ← t({C_t}ch,{H_t}px)")
            else:
                desc = (f"t({C_t}ch,{H_t}px) → bottleneck(r={rank}) → ({C_s}ch,{H_t}px) [target]  "
                        f"vs  s({C_s}ch,{H_s}px) → pool → ({C_s}ch,{H_t}px)")

            print(f"  L{lvl}: {desc}\n"
                  f"        params: {bneck_p:,}  (direct 1×1 would be {direct_p:,})")

        reduction = 100 * (1 - total_bottleneck / total_direct)
        print(f"  Total projector params: {total_bottleneck:,}  "
              f"(direct 1×1 total would be {total_direct:,}, -{reduction:.0f}%)\n")

        # ── Auto-lambda buffers ───────────────────────────────────────────────
        self.register_buffer("_cosine_scale", torch.tensor(1.0))
        self.register_buffer("_at_scale",     torch.tensor(1.0))
        self.register_buffer("_rkd_scale",    torch.tensor(1.0))
        self.register_buffer("_scales_set",   torch.tensor(False))

        # ── teacher_to_student: warmup tracking ───────────────────────────────
        # NOTE: we deliberately do NOT toggle requires_grad on student/projector
        # parameters here. TorchDynamo does not guard on a Parameter's
        # requires_grad flag, so toggling it after `module.student =
        # torch.compile(module.student)` produces a stale compiled graph:
        # the student's forward keeps treating its own weights as constants
        # even after requires_grad is set back to True, and loss.backward()
        # fails with "does not require grad and does not have a grad_fn".
        # Gradient flow is instead controlled via .detach() on the relevant
        # OUTPUT tensor inside _cosine_loss — pure Python control flow on
        # self.current_epoch, invisible to the compiled student submodule.

    def on_train_epoch_start(self) -> None:
        if self.hparams.proj_dir != "teacher_to_student":
            return

        warmup_active = self.current_epoch < self.hparams.projector_warmup_epochs

        # BN running-stats freeze is safe with torch.compile: switching
        # self.training via .train()/.eval() IS a guard Dynamo supports
        # correctly (unlike requires_grad), so this recompiles cleanly.
        self.projectors.train(warmup_active)

        if self.current_epoch == 0 and warmup_active:
            print(f"[KDModule] teacher_to_student: projector warmup for "
                  f"{self.hparams.projector_warmup_epochs} epoch(s) "
                  f"(student gradient detached).")
        elif self.current_epoch == self.hparams.projector_warmup_epochs:
            print("[KDModule] teacher_to_student: warmup done — "
                  "student now trains against frozen projector targets.")

    # ── Cosine loss ───────────────────────────────────────────────────────────

    def _cosine_loss(
        self,
        s_feats: list[torch.Tensor],
        t_feats: list[torch.Tensor],
        prefix: str,
    ) -> torch.Tensor:
        total = s_feats[0].new_zeros(())

        # Pure Python condition — invisible to the compiled student submodule.
        # See on_train_epoch_start for why this replaces requires_grad toggling.
        warmup_active = (
            prefix == "train"
            and self.hparams.proj_dir == "teacher_to_student"
            and self.current_epoch < self.hparams.projector_warmup_epochs
        )

        for lvl, (s_f, t_f, proj, w) in enumerate(
            zip(s_feats, t_feats, self.projectors, self.level_weights)
        ):
            # Pool student to teacher spatial resolution (both directions)
            s_pooled = F.adaptive_avg_pool2d(s_f, t_f.shape[-2:])

            if self.hparams.proj_dir == "student_to_teacher":
                s_proj = proj(s_pooled)                   # (B, C_t, H_t, W_t)
                a = F.normalize(s_proj.flatten(2), p=2, dim=1)
                b = F.normalize(t_f.flatten(2),    p=2, dim=1)

            else:
                # teacher_to_student: project teacher → student channel space
                t_proj = proj(t_f)                        # (B, C_s, H_t, W_t)

                if warmup_active:
                    # Phase 1: train the projector only.
                    # Gradient flows into `proj`, NOT into the student backbone.
                    a = F.normalize(s_pooled.detach().flatten(2), p=2, dim=1)
                    b = F.normalize(t_proj.flatten(2),            p=2, dim=1)
                else:
                    # Phase 2: projector target is now fixed (BN in eval mode,
                    # see on_train_epoch_start). Gradient flows into the
                    # student backbone only.
                    a = F.normalize(s_pooled.flatten(2),       p=2, dim=1)
                    b = F.normalize(t_proj.detach().flatten(2), p=2, dim=1)

            cos_l = (1.0 - (a * b).sum(dim=1)).mean()
            total = total + w * cos_l
            self.log(f"{prefix}/cos_l{lvl}", cos_l,
                     on_step=False, on_epoch=True, sync_dist=True)

        return total / sum(self.level_weights)

    # ── Combined loss with auto-lambda ────────────────────────────────────────

    def _total_loss(
        self,
        s_feats: list[torch.Tensor],
        t_feats: list[torch.Tensor],
        prefix: str,
    ) -> torch.Tensor:
        losses: dict[str, torch.Tensor] = {}

        if self.hparams.use_cosine:
            losses["cosine"] = self._cosine_loss(s_feats, t_feats, prefix)

        if self.hparams.use_at:
            at_total, at_per_lvl = _at_loss(s_feats, t_feats, self.level_weights)
            losses["at"] = at_total
            for lvl, l in enumerate(at_per_lvl):
                self.log(f"{prefix}/at_l{lvl}", l,
                         on_step=False, on_epoch=True, sync_dist=True)

        if self.hparams.use_rkd:
            losses["rkd"] = _rkd_loss(s_feats[-1], t_feats[-1])
            self.log(f"{prefix}/rkd", losses["rkd"],
                     on_step=False, on_epoch=True, sync_dist=True)

        # Set auto-lambda scales at the first training step
        if prefix == "train" and not self._scales_set:
            if "cosine" in losses:
                self._cosine_scale.copy_(losses["cosine"].detach().clamp(min=1e-6))
            if "at" in losses:
                self._at_scale.copy_(losses["at"].detach().clamp(min=1e-6))
            if "rkd" in losses:
                self._rkd_scale.copy_(losses["rkd"].detach().clamp(min=1e-6))
            self._scales_set.fill_(True)

        scales = {
            "cosine": self._cosine_scale,
            "at":     self._at_scale,
            "rkd":    self._rkd_scale,
        }
        total = sum(v / scales[k] for k, v in losses.items()) / len(losses)

        for k, v in losses.items():
            self.log(f"{prefix}/{k}", v, on_step=False, on_epoch=True, sync_dist=True)

        return total

    # ── Lightning steps ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.student(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # Resize S2 images to 10m resolution before teacher forward
        img_s = batch["simulated"]                            # 4.75m → 224px
        img_t = batch["sentinel2"]                            # already 10m → ~106px

        with torch.no_grad():
            self.teacher.eval()
            t_feats = self.teacher(img_t)

        s_feats = self.student(img_s)
        loss    = self._total_loss(s_feats, t_feats, "train")

        self.log("train_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        img_s = batch["simulated"]
        img_t = batch["sentinel2"]
        if img_t.shape[-1] != int(img_s.shape[-1] * 0.475):
            img_t = F.interpolate(
                img_t,
                size=(int(img_s.shape[-2] * 0.475), int(img_s.shape[-1] * 0.475)),
                mode="bilinear", align_corners=False,
            )

        with torch.no_grad():
            t_feats = self.teacher(img_t)
            s_feats = self.student(img_s)

        loss = self._total_loss(s_feats, t_feats, "val")
        self.log("val_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=img_s.shape[0])

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        if self.hparams.proj_dir == "student_to_teacher":
            # Projector from random init → needs faster adaptation
            proj_lr = self.hparams.lr * 10.0
        else:
            # Projector warms up at high LR during Phase 1, student at base LR
            proj_lr = self.hparams.lr * 10.0

        param_groups = [
            {"params": self.student.parameters(),
             "lr": self.hparams.lr, "name": "student"},
            {"params": self.projectors.parameters(),
             "lr": proj_lr,         "name": "projectors"},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.hparams.weight_decay
        )

        warmup_epochs = self.hparams.warmup_epochs
        cosine_epochs = max(1, self.trainer.max_epochs - warmup_epochs)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, cosine], milestones=[warmup_epochs],
            ),
        }