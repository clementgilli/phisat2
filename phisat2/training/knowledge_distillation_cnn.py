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
    """Attention Transfer (Zagoruyko & Komodakis, 2016)."""
    per_level: list[torch.Tensor] = []
    total = s_feats[0].new_zeros(())
    for s_f, t_f, w in zip(s_feats, t_feats, weights):
        s_attn = _attention_map(s_f)
        t_attn = _attention_map(t_f)
        if s_attn.shape[-2:] != t_attn.shape[-2:]:
            s_attn = F.adaptive_avg_pool2d(s_attn.unsqueeze(1), t_attn.shape[-2:]).squeeze(1)
        lvl = F.mse_loss(s_attn, t_attn.detach())
        per_level.append(lvl)
        total = total + w * lvl
    return total / sum(weights), per_level


def _rkd_loss(s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
    """Relational KD — distance-wise (Park et al., CVPR 2019)."""
    s = F.adaptive_avg_pool2d(s_feat, 1).flatten(1)
    t = F.adaptive_avg_pool2d(t_feat, 1).flatten(1)
    s_d = torch.cdist(s, s, p=2)
    t_d = torch.cdist(t, t, p=2)
    s_d = s_d / (s_d.mean() + 1e-6)
    t_d = t_d / (t_d.mean() + 1e-6)
    return F.smooth_l1_loss(s_d, t_d.detach())


def _infonce_loss(z_s: torch.Tensor, z_t: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE symmetric cross-entropy on L2-normalised projections."""
    B = z_s.shape[0]
    logits = z_s @ z_t.T / temperature
    labels = torch.arange(B, device=z_s.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Projection heads
# ─────────────────────────────────────────────────────────────────────────────

class InfoNCEProjector(nn.Module):
    """GAP → Linear → BN → GELU → Linear → BN → L2-norm."""
    def __init__(self, in_channels: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Linear(in_channels, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = self.gap(x).flatten(1)
        return F.normalize(self.proj(x), p=2, dim=1)


class BottleneckProjector(nn.Module):
    """Low-rank Conv1×1 projector: C_in → r → C_out."""
    def __init__(self, c_in: int, c_out: int, rank: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(c_in, rank, 1, bias=False), nn.BatchNorm2d(rank), nn.GELU())
        self.expand = nn.Sequential(nn.Conv2d(rank, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(self.reduce(x))


class MGDHead(nn.Module):
    """
    Masked Generative Distillation head — one per encoder level.
    (Yang et al., ECCV 2022 — exact architecture from eq. 3-5)

    Architecture (from the paper):
        f_align : Conv1×1(C_s → C_t) + BN  — channel alignment (eq. 3)
                  Can be replaced by BottleneckProjector when C_s*C_t is large.
        mask    : λ% of spatial positions zeroed randomly
        G       : W_12(ReLU(W_11(·)))       — two Conv3×3 (eq. 4)
                  Generates teacher features from the masked aligned features.
        loss    : MSE(G(f_align(S)·M), T)   (eq. 5)

    Note: with S2 upsampled to match PhiSat-2 resolution (both 224×224),
    student and teacher features are at the same spatial resolution at each
    level — no spatial pooling needed, no information destroyed.

    Args
    ----
    student_channels : C_s
    teacher_channels : C_t
    mask_ratio       : λ in the paper (default 0.75)
    use_bottleneck   : replace f_align Conv1×1 with BottleneckProjector
                       (useful when C_s or C_t is large, e.g. 512+ channels)
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        mask_ratio: float = 0.75,
        use_bottleneck: bool = False,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio

        # f_align: Conv1×1 (or low-rank bottleneck) — channel alignment only
        if use_bottleneck:
            rank = _default_rank(student_channels, teacher_channels)
            self.f_align: nn.Module = BottleneckProjector(student_channels, teacher_channels, rank)
        else:
            self.f_align = nn.Sequential(
                nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(teacher_channels),
            )

        # G = W_12(ReLU(W_11(·))): two Conv3×3 operating in teacher channel space
        self.G = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """
        s_feat : (B, C_s, H, W)
        t_feat : (B, C_t, H, W)  — same spatial res when S2 upsampled to PhiSat-2 res
        """
        # Step 1: f_align — channel alignment (eq. 3, left part)
        s_aligned = self.f_align(s_feat)                           # (B, C_t, H, W)

        # Step 2: random spatial mask (broadcast over channels)
        mask = (torch.rand(s_aligned.shape[0], 1, *s_aligned.shape[2:],
                           device=s_aligned.device) > self.mask_ratio).float()
        s_masked = s_aligned * mask                                 # (B, C_t, H, W)

        # Step 3: G — generate teacher features from masked aligned features (eq. 4)
        s_gen = self.G(s_masked)                                    # (B, C_t, H, W)

        # Step 4: spatial alignment if still needed (no-op when same resolution)
        if s_gen.shape[-2:] != t_feat.shape[-2:]:
            s_gen = F.adaptive_avg_pool2d(s_gen, t_feat.shape[-2:])

        # Step 5: MSE loss (eq. 5)
        return F.mse_loss(s_gen, t_feat.detach())


def _default_rank(c_in: int, c_out: int, min_rank: int = 16, divisor: int = 4) -> int:
    r = max(min_rank, min(c_in, c_out) // divisor)
    return min(r, c_in, c_out)


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────

class CNNKDModule(L.LightningModule):
    """
    CNN→CNN multi-scale feature distillation.

    Loss modes (mutually exclusive with use_mgd):
        use_cosine — cosine alignment via low-rank projector
        use_at     — attention map transfer (no projector)
        use_rkd    — pairwise distance structure on deepest level (no projector)
        use_infonce — contrastive cross-sensor alignment
        use_mgd    — Masked Generative Distillation (Yang et al., ECCV 2022)
                     Disables all other losses when True.

    Auto-lambda:
        Each active loss is normalised by its value at the first training step
        → all terms start at 1.0 without manual λ tuning.
        InfoNCE scale is now correctly included.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        spec: Any,
        *,
        proj_dir: Literal["student_to_teacher", "teacher_to_student"] = "student_to_teacher",
        use_cosine:  bool  = False,
        use_at:      bool  = False,
        use_rkd:     bool  = False,
        use_infonce: bool  = False,
        use_mgd:     bool  = True,
        infonce_proj_dim:    int   = 256,
        infonce_temperature: float = 0.07,
        mgd_mask_ratio:      float = 0.75,
        lr:                      float       = 1e-4,
        weight_decay:            float       = 1e-4,
        level_weights:           list[float] | None = None,
        warmup_epochs:           int         = 5,
        projector_warmup_epochs: int         = 1,
        projector_rank:          int | None  = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["student_model", "teacher_model"])

        # MGD is exclusive — ignore other loss flags when set
        if use_mgd:
            if use_cosine or use_at or use_rkd or use_infonce:
                print("[CNNKDModule] use_mgd=True → disabling cosine / at / rkd / infonce.")
        elif not (use_cosine or use_at or use_rkd or use_infonce):
            raise ValueError("Enable at least one loss flag.")

        self.student = student_model
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        _DUMMY_S = 224
        _DUMMY_T = 224 #int(_DUMMY_S * 0.475)
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

        # ── Cosine projectors ─────────────────────────────────────────────────
        self.projectors = nn.ModuleList()
        if not use_mgd:
            print(f"\n[CNNKDModule] proj_dir={proj_dir}  "
                  f"cosine={use_cosine} at={use_at} rkd={use_rkd} infonce={use_infonce}")
            total_direct = total_bneck = 0
            for lvl, (s_f, t_f) in enumerate(zip(s_feats, t_feats)):
                C_s, H_s = s_f.shape[1], s_f.shape[2]
                C_t, H_t = t_f.shape[1], t_f.shape[2]
                c_in, c_out = (C_s, C_t) if proj_dir == "student_to_teacher" else (C_t, C_s)
                if c_in == c_out:
                    proj = nn.Identity()
                    dp = bp = 0
                    desc = f"s({C_s}ch,{H_s}px) == t({C_t}ch,{H_t}px)  [identity]"
                else:
                    rank = projector_rank or _default_rank(c_in, c_out)
                    proj = BottleneckProjector(c_in, c_out, rank)
                    dp = c_in * c_out
                    bp = c_in * rank + rank * c_out
                    desc = f"s({C_s}ch,{H_s}px) → r={rank} → t({C_t}ch,{H_t}px)"
                self.projectors.append(proj)
                total_direct += dp; total_bneck += bp
                print(f"  L{lvl}: {desc}" + (f" | params: {bp:,}" if bp else ""))
            if total_direct:
                print(f"  Total projector params: {total_bneck:,} (-{100*(1-total_bneck/total_direct):.0f}% vs direct)\n")

        # ── InfoNCE heads ─────────────────────────────────────────────────────
        self.s_infonce_proj = nn.ModuleList()
        self.t_infonce_proj = nn.ModuleList()
        if use_infonce and not use_mgd:
            for s_f, t_f in zip(s_feats, t_feats):
                self.s_infonce_proj.append(InfoNCEProjector(s_f.shape[1], infonce_proj_dim))
                self.t_infonce_proj.append(InfoNCEProjector(t_f.shape[1], infonce_proj_dim))
            print(f"[CNNKDModule] InfoNCE: proj_dim={infonce_proj_dim}, τ={infonce_temperature}")

        # ── MGD heads ─────────────────────────────────────────────────────────
        self.mgd_heads = nn.ModuleList()
        if use_mgd:
            for lvl, (s_f, t_f) in enumerate(zip(s_feats, t_feats)):
                C_s, H_s = s_f.shape[1], s_f.shape[2]
                C_t, H_t = t_f.shape[1], t_f.shape[2]
                head = MGDHead(C_s, C_t, mask_ratio=mgd_mask_ratio, use_bottleneck=True)
                self.mgd_heads.append(head)
                n_params = sum(p.numel() for p in head.parameters())
                print(f"[CNNKDModule] L{lvl}: s({C_s}ch,{H_s}px) → mask(λ={mgd_mask_ratio}) "
                      f"→ G → t({C_t}ch,{H_t}px) | params: {n_params:,}")

        # ── Auto-lambda buffers ───────────────────────────────────────────────
        for key in ("cosine", "at", "rkd", "infonce", "mgd"):
            self.register_buffer(f"_scale_{key}", torch.tensor(1.0))
        self.register_buffer("_scales_set", torch.tensor(False))
        
        self.test_step_outputs: list[torch.Tensor] = []

    # ── teacher_to_student warmup ─────────────────────────────────────────────

    def on_train_epoch_start(self) -> None:
        if self.hparams.proj_dir != "teacher_to_student" or self.hparams.use_mgd:
            return
        warmup_active = self.current_epoch < self.hparams.projector_warmup_epochs
        self.projectors.train(warmup_active)
        if self.current_epoch == 0 and warmup_active:
            print(f"[CNNKDModule] projector warmup for {self.hparams.projector_warmup_epochs} epoch(s).")
        elif self.current_epoch == self.hparams.projector_warmup_epochs:
            print("[CNNKDModule] projector frozen — student now trains against fixed targets.")

    # ── Individual losses ─────────────────────────────────────────────────────

    def _cosine_loss(self, s_feats, t_feats, prefix):
        total = s_feats[0].new_zeros(())
        warmup_active = (
            prefix == "train"
            and self.hparams.proj_dir == "teacher_to_student"
            and self.current_epoch < self.hparams.projector_warmup_epochs
        )
        for lvl, (s_f, t_f, proj, w) in enumerate(
            zip(s_feats, t_feats, self.projectors, self.level_weights)
        ):
            s_pooled = F.adaptive_avg_pool2d(s_f, t_f.shape[-2:])
            if self.hparams.proj_dir == "student_to_teacher":
                s_proj = proj(s_pooled)
                a = F.normalize(s_proj.flatten(2), p=2, dim=1)
                b = F.normalize(t_f.flatten(2),    p=2, dim=1)
            else:
                t_proj = proj(t_f)
                if warmup_active:
                    a = F.normalize(s_pooled.detach().flatten(2), p=2, dim=1)
                    b = F.normalize(t_proj.flatten(2),            p=2, dim=1)
                else:
                    a = F.normalize(s_pooled.flatten(2),        p=2, dim=1)
                    b = F.normalize(t_proj.detach().flatten(2), p=2, dim=1)
            cos_l = (1.0 - (a * b).sum(dim=1)).mean()
            total = total + w * cos_l
            self.log(f"{prefix}/cos_l{lvl}", cos_l, on_step=False, on_epoch=True, sync_dist=True)
        return total / sum(self.level_weights)

    def _mgd_loss(self, s_feats, t_feats, prefix):
        """MGD loss: sum over levels with level_weights (eq. 5 in the paper)."""
        total = s_feats[0].new_zeros(())
        for lvl, (s_f, t_f, head, w) in enumerate(
            zip(s_feats, t_feats, self.mgd_heads, self.level_weights)
        ):
            lvl_loss = head(s_f, t_f)
            total = total + w * lvl_loss
            self.log(f"{prefix}/mgd_l{lvl}", lvl_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total / sum(self.level_weights)

    # ── Combined loss with auto-lambda ────────────────────────────────────────

    def _total_loss(self, s_feats, t_feats, prefix):
        losses: dict[str, torch.Tensor] = {}

        if self.hparams.use_mgd:
            losses["mgd"] = self._mgd_loss(s_feats, t_feats, prefix)
        else:
            if self.hparams.use_cosine:
                losses["cosine"] = self._cosine_loss(s_feats, t_feats, prefix)
            if self.hparams.use_at:
                at_total, at_per_lvl = _at_loss(s_feats, t_feats, self.level_weights)
                losses["at"] = at_total
                for lvl, l in enumerate(at_per_lvl):
                    self.log(f"{prefix}/at_l{lvl}", l, on_step=False, on_epoch=True, sync_dist=True)
            if self.hparams.use_rkd:
                losses["rkd"] = _rkd_loss(s_feats[-1], t_feats[-1])
                self.log(f"{prefix}/rkd", losses["rkd"], on_step=False, on_epoch=True, sync_dist=True)
            if self.hparams.use_infonce:
                infonce_total = s_feats[0].new_zeros(())
                for lvl, (s_f, t_f, s_proj, t_proj, w) in enumerate(zip(
                    s_feats, t_feats, self.s_infonce_proj, self.t_infonce_proj, self.level_weights
                )):
                    z_s = s_proj(s_f)
                    z_t = t_proj(t_f.detach())
                    lvl_loss = _infonce_loss(z_s, z_t, self.hparams.infonce_temperature)
                    infonce_total = infonce_total + w * lvl_loss
                    self.log(f"{prefix}/infonce_l{lvl}", lvl_loss, on_step=False, on_epoch=True, sync_dist=True)
                losses["infonce"] = infonce_total / sum(self.level_weights)

        # Auto-lambda: record initial scale once at the first training step
        if prefix == "train" and not self._scales_set:
            for k, v in losses.items():
                getattr(self, f"_scale_{k}").copy_(v.detach().clamp(min=1e-6))
            self._scales_set.fill_(True)

        total = sum(v / getattr(self, f"_scale_{k}") for k, v in losses.items()) / len(losses)

        for k, v in losses.items():
            self.log(f"{prefix}/{k}", v, on_step=False, on_epoch=True, sync_dist=True)

        return total

    # ── Lightning steps ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.student(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        img_s = batch["simulated"]
        img_t = batch["sentinel2"]
        
        with torch.no_grad():
            self.teacher.eval()
            t_feats = self.teacher(img_t)

        s_feats = self.student(img_s)
        loss    = self._total_loss(s_feats, t_feats, "train")

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        img_s = batch["simulated"]
        img_t = batch["sentinel2"]

        with torch.no_grad():
            t_feats = self.teacher(img_t)
            s_feats = self.student(img_s)

        loss = self._total_loss(s_feats, t_feats, "val")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=img_s.shape[0])

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        lr = self.hparams.lr
        head_lr = lr * 10.0   # all training-only heads start from random init

        param_groups = [
            {"params": self.student.parameters(), "lr": lr, "name": "student"},
        ]

        if self.hparams.use_mgd:
            param_groups.append(
                {"params": self.mgd_heads.parameters(), "lr": head_lr, "name": "mgd_heads"}
            )
        else:
            param_groups.append(
                {"params": self.projectors.parameters(), "lr": head_lr, "name": "projectors"}
            )
            if self.hparams.use_infonce:
                param_groups.append(
                    {"params": list(self.s_infonce_proj.parameters()) +
                               list(self.t_infonce_proj.parameters()),
                     "lr": head_lr, "name": "infonce_heads"}
                )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)

        warmup_epochs = self.hparams.warmup_epochs
        cosine_epochs = max(1, self.trainer.max_epochs - warmup_epochs)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, cosine], milestones=[warmup_epochs]
            ),
        }
        
    def test_step(self, batch: dict, batch_idx: int) -> None:
        img_s = batch["simulated"]
        img_t = batch["sentinel2"]

        with torch.no_grad():
            self.teacher.eval()
            t_feats = self.teacher(img_t)
            s_feats = self.student(img_s)

        batch_matrix = torch.zeros(self.n_levels, len(t_feats), device=self.device)
        
        for i, s_f in enumerate(s_feats):
            for j, t_f in enumerate(t_feats):
                batch_matrix[i, j] = self._compute_linear_cka(s_f, t_f)
                
        self.test_step_outputs.append(batch_matrix)

    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs:
            return

        cka_matrix = torch.stack(self.test_step_outputs).mean(dim=0).cpu().numpy()
        self.test_step_outputs.clear()
        
        print(f"\n[CKA MATRIX] Layer-by-Layer Representation Alignment")
        print(f"Teacher: {self.hparams.spec if hasattr(self.hparams, 'spec') else 'Frozen Teacher'}")
        print("Rows = Student Layers (0 à N), Cols = Teacher Layers (0 à M)\n")
        print(cka_matrix)
        print("\n" + "="*60 + "\n")

        out_dir = Path(self.trainer.default_root_dir) / "cka_matrix.csv"
        
        df = pd.DataFrame(
            cka_matrix,
            index=[f"student_enc_{i}" for i in range(cka_matrix.shape[0])],
            columns=[f"teacher_enc_{j}" for j in range(cka_matrix.shape[1])]
        )
        df.to_csv(out_dir)
        print(f"[CNNKDModule] CKA Matrix successfully saved to {out_dir}")
        
    @torch.no_grad()
    def _compute_linear_cka(self, feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        X = feat_s.flatten(start_dim=1).float()
        Y = feat_t.flatten(start_dim=1).float()

        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        K = X @ X.T
        L = Y @ Y.T

        cross_term = torch.sum(K * L)
        norm_K = torch.sqrt(torch.sum(K * K))
        norm_L = torch.sqrt(torch.sum(L * L))

        return cross_term / (norm_K * norm_L + 1e-8)