from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# GL Projector
# ─────────────────────────────────────────────────────────────────────────────

class GLProjector(nn.Module):
    """
    Group-wise Linear Projector (CAKD — Liu et al., ACCV 2022, eq. 5-6).

    Maps a CNN spatial feature map to the ViT patch-token space in a
    pixel-by-pixel manner, without the information loss of global pooling.

    Nearby spatial positions share an FC layer (4×4 groups → 16 FC layers
    for a 14×14 patch grid) instead of N=196 independent layers.

    Pipeline: (B, C_s, H_s, W_s)
        → AdaptiveAvgPool → (B, C_s, H_p, W_p)
        → group-wise FC   → (B, H_p, W_p, D)
        → flatten         → (B, N, D)          no L2 norm (not in the paper)
    """

    def __init__(
        self,
        in_channels:  int,
        vit_dim:      int,
        patch_grid:   tuple[int, int] = (14, 14),
        group_size:   int = 4,
    ) -> None:
        super().__init__()
        self.patch_grid = patch_grid
        self.group_size = group_size
        H_p, W_p       = patch_grid

        self.n_groups_h = math.ceil(H_p / group_size)
        self.n_groups_w = math.ceil(W_p / group_size)

        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, vit_dim, bias=False),
                nn.LayerNorm(vit_dim),
            )
            for _ in range(self.n_groups_h * self.n_groups_w)
        ])

        n_params = sum(p.numel() for m in self.fc for p in m.parameters())
        print(f"[GLProjector] {in_channels}ch → {vit_dim}d | "
              f"patch_grid={patch_grid} group={group_size}×{group_size} | "
              f"groups={len(self.fc)} | params={n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_s, H_s, W_s) → (B, N, D)"""
        H_p, W_p = self.patch_grid
        if x.shape[-2:] != (H_p, W_p):
            x = F.adaptive_avg_pool2d(x, (H_p, W_p))

        x = x.permute(0, 2, 3, 1)  # (B, H_p, W_p, C_s)

        out = torch.empty(x.shape[0], H_p, W_p, self.fc[0][0].out_features,
                          device=x.device, dtype=x.dtype)
        g = 0
        for gh in range(self.n_groups_h):
            for gw in range(self.n_groups_w):
                h0 = gh * self.group_size;  h1 = min(h0 + self.group_size, H_p)
                w0 = gw * self.group_size;  w1 = min(w0 + self.group_size, W_p)
                out[:, h0:h1, w0:w1, :] = self.fc[g](x[:, h0:h1, w0:w1, :])
                g += 1

        return out.flatten(1, 2)   # (B, N, D)  — no L2 norm (not in the paper)


# ─────────────────────────────────────────────────────────────────────────────
# PCA Projector
# ─────────────────────────────────────────────────────────────────────────────

class PCAProjector(nn.Module):
    """
    Partially Cross Attention Projector (CAKD — Liu et al., ACCV 2022, eq. 1-4).

    Maps student CNN features to Q_S, K_S, V_S via three Conv3×3 layers,
    then computes a "partial" self-attention where student Q/K/V are randomly
    replaced by teacher Q/K/V (p=0.5), forcing the student to mimic the
    teacher's attention structure.

    Teacher Q/K/V are captured via a forward hook on the last teacher QKV
    linear layer (registered in ViTKDModule.__init__).  The hook output is
    (B, N, 3·D); this class splits it into Q_t, K_t, V_t.

    Loss (eq. 4):
        L_pca = ||Attn_T − PCAttn_S||² + ||V_T·V_T^T/√d − V_S·V_S^T/√d||²

    Args
    ----
    in_channels : C_s  — student bottleneck channels.
    vit_dim     : D    — ViT embedding dimension (= QKV_out / 3).
    patch_grid  : (H_p, W_p) spatial token grid (14×14 for ViT/16 on 224×224).
    """

    def __init__(
        self,
        in_channels: int,
        vit_dim:     int,
        patch_grid:  tuple[int, int] = (14, 14),
    ) -> None:
        super().__init__()
        self.D          = vit_dim
        self.patch_grid = patch_grid

        # Three independent Conv3×3 → one per Q/K/V
        def _proj_head():
            return nn.Sequential(
                nn.Conv2d(in_channels, vit_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(vit_dim),
            )
        self.conv_q = _proj_head()
        self.conv_k = _proj_head()
        self.conv_v = _proj_head()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[PCAProjector] {in_channels}ch → 3×{vit_dim}d (Q/K/V) | "
              f"patch_grid={patch_grid} | params={n_params:,}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_tokens(self, feat_map: torch.Tensor) -> torch.Tensor:
        """(B, D, H, W) → (B, N, D) aligned to patch grid."""
        H_p, W_p = self.patch_grid
        if feat_map.shape[-2:] != (H_p, W_p):
            feat_map = F.adaptive_avg_pool2d(feat_map, (H_p, W_p))
        return feat_map.flatten(2).transpose(1, 2)   # (B, N, D)

    @staticmethod
    def _partial_mix(M_s: torch.Tensor, M_t: torch.Tensor) -> torch.Tensor:
        """Element-wise random mix: each element uses M_t with p=0.5."""
        mask = torch.rand_like(M_s) >= 0.5
        return torch.where(mask, M_t, M_s)

    @staticmethod
    def _attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention. Returns (attn_map, V_out)."""
        d    = Q.shape[-1]
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d), dim=-1)
        return attn, V   # we only use the attn map and raw V for the loss

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        s_feat: torch.Tensor,   # (B, C_s, H_s, W_s) student bottleneck
        qkv_t:  torch.Tensor,   # (B, N, 3·D) raw QKV from teacher hook
    ) -> torch.Tensor:
        """Returns scalar L_pca loss."""
        D = self.D

        # Split teacher QKV
        Q_t = qkv_t[:, :,  :D ].detach()
        K_t = qkv_t[:, :, D:2*D].detach()
        V_t = qkv_t[:, :, 2*D: ].detach()

        # Teacher attention map
        Attn_t, _ = self._attn(Q_t, K_t, V_t)   # (B, N, N)

        # Student Q/K/V via Conv3×3
        Q_s = self._to_tokens(self.conv_q(s_feat))   # (B, N, D)
        K_s = self._to_tokens(self.conv_k(s_feat))
        V_s = self._to_tokens(self.conv_v(s_feat))

        # Partial cross attention (stochastic mix at training, pure student at eval)
        if self.training:
            Q_pc = self._partial_mix(Q_s, Q_t)
            K_pc = self._partial_mix(K_s, K_t)
            V_pc = self._partial_mix(V_s, V_t)
        else:
            Q_pc, K_pc, V_pc = Q_s, K_s, V_s

        PCAttn_s, _ = self._attn(Q_pc, K_pc, V_pc)   # (B, N, N)

        # Loss 1: attention map alignment
        L_attn = F.mse_loss(PCAttn_s, Attn_t)

        # Loss 2: value correlation alignment (eq. 4, second term)
        d      = float(D)
        Vc_t   = torch.bmm(V_t, V_t.transpose(1, 2)) / math.sqrt(d)
        Vc_s   = torch.bmm(V_s, V_s.transpose(1, 2)) / math.sqrt(d)
        L_val  = F.mse_loss(Vc_s, Vc_t)

        return L_attn + L_val


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    3-FC-layer adversarial discriminator (CAKD, eq. 8-9).

    Distinguishes real teacher tokens (S2 → TerraMind) from projected
    student tokens (PhiSat-2 → PhiSatNet → GL projector).  The sensor
    gap between S2 and PhiSat-2 sim plays the role of the MVG in the
    original paper, without requiring synthetic augmentations.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(dim // 2, dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, D) or (B, D) → (B, 1) logit."""
        if tokens.ndim == 3:
            tokens = tokens.mean(dim=1)
        return self.net(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# ViT→CNN KD Module
# ─────────────────────────────────────────────────────────────────────────────

class ViTKDModule(L.LightningModule):
    """
    ViT teacher → CNN student knowledge distillation.

    Implements GL + PCA projectors from CAKD (Liu et al., ACCV 2022) with an
    adversarial discriminator adapted to the cross-sensor satellite setting.

    Losses
    ------
    L_gl   = MSE(GL(s_bottleneck), t_tokens)          — GL projector alignment
    L_pca  = ||Attn_T − PCAttn_S||² + ||Vc_T − Vc_S||²  — PCA attention alignment
    L_adv  = BCE(D(GL(s)), ones)                        — student fools discriminator
    L_disc = BCE(D(t), 1) + BCE(D(GL(s).detach()), 0)  — discriminator

    Total (student):  L_gl + λ_pca·L_pca + λ_adv·L_adv
    Total (disc):     L_disc  (updated every disc_update_freq student steps)

    Teacher QKV hook
    ----------------
    When use_pca=True, a forward hook is registered on the LAST 'qkv' linear
    layer found in the teacher.  The hook captures the raw (B, N, 3·D) output
    and stores it in self._qkv_cache, which is consumed by PCAProjector after
    each teacher forward pass.  Generic: works for any ViT with a linear layer
    named *qkv* in its module hierarchy.

    Natural MVG
    -----------
    No Multi-View Generator needed: teacher sees S2, student sees PhiSat-2 sim
    of the same scene.  The sensor gap is already a harder discrimination task
    than any synthetic augmentation used in the original paper.

    Args
    ----
    use_pca       : Enable PCA projector (requires QKV hook on teacher).
    lambda_pca    : Weight of L_pca relative to L_gl.
    lambda_adv    : Weight of adversarial loss.
    disc_update_freq : Update discriminator every N student steps.
    """

    automatic_optimization: bool = False   # GAN-style alternating updates

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        spec: Any,
        *,
        patch_grid:        tuple[int, int] = (14, 14),
        gl_group_size:     int   = 4,
        use_pca:           bool  = True,
        lambda_pca:        float = 1.0,
        lambda_adv:        float = 0.1,
        disc_update_freq:  int   = 5,
        lr:                float = 1e-4,
        weight_decay:      float = 1e-4,
        warmup_epochs:     int   = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["student_model", "teacher_model"])

        self.student = student_model
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # ── Dummy forward to infer shapes ─────────────────────────────────────
        _D = 224
        with torch.no_grad():
            s_feats = self.student(torch.zeros(1, 8,  _D, _D))
            t_out   = self.teacher(torch.zeros(1, 13, _D, _D))

        t_tokens = t_out["patch_tokens"]
        assert t_tokens.ndim == 3, (
            f"Expected ViT token sequence (B, N, D), got {t_tokens.shape}."
        )
        N, D = t_tokens.shape[1], t_tokens.shape[2]
        C_s  = s_feats[-1].shape[1]

        print(f"\n[ViTKDModule] student bottleneck: ({C_s}ch, {s_feats[-1].shape[-1]}px)"
              f"  →  GL+PCA  →  teacher tokens: ({N}tok, {D}d)"
              f"  use_pca={use_pca}")

        # ── GL Projector ──────────────────────────────────────────────────────
        self.gl_proj = GLProjector(C_s, D, patch_grid, gl_group_size)

        # ── PCA Projector + QKV hook ──────────────────────────────────────────
        self._qkv_cache:    torch.Tensor | None = None
        self._hook_handle:  Any                 = None

        if use_pca:
            self.pca_proj = PCAProjector(C_s, D, patch_grid)
            self._hook_handle = self._register_qkv_hook()
        else:
            self.pca_proj = None  # type: ignore[assignment]

        # ── Discriminator ─────────────────────────────────────────────────────
        self.discriminator = Discriminator(D)

        # ── Step counter for discriminator updates ────────────────────────────
        self._step_count = 0

        # ── Auto-lambda scale buffers ─────────────────────────────────────────
        self.register_buffer("_scale_gl",  torch.tensor(1.0))
        self.register_buffer("_scale_pca", torch.tensor(1.0))
        self.register_buffer("_scales_set", torch.tensor(False))

    # ── QKV hook ─────────────────────────────────────────────────────────────

    def _register_qkv_hook(self):
        """
        Register a forward hook on the last 'qkv' linear layer of the teacher.
        Generic: searches named_modules() for any Linear with 'qkv' in its name.
        The hook automatically strips special tokens (like [CLS]) if present,
        storing only the spatial tokens (B, H*W, 3·D) in self._qkv_cache.
        """
        last_name, last_mod = None, None
        for name, mod in self.teacher.named_modules():
            if "qkv" in name.lower() and isinstance(mod, nn.Linear):
                last_name, last_mod = name, mod

        if last_mod is None:
            raise RuntimeError(
                "use_pca=True but no 'qkv' Linear layer found in teacher. "
                "Check the teacher architecture."
            )
        print(f"[ViTKDModule] QKV hook → {last_name}  (out_features={last_mod.out_features})")

        def _hook(mod, inp, out):
            # out shape: (B, N, 3*D)
            B, N, dim = out.shape
            grid_size = int(math.isqrt(N))
            
            if grid_size * grid_size == N:
                self._qkv_cache = out
            else:
                num_special = N - (grid_size * grid_size)
                self._qkv_cache = out[:, num_special:, :]

        return last_mod.register_forward_hook(_hook)

    def on_fit_end(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()

    # ── Forward helpers ───────────────────────────────────────────────────────

    def _teacher_tokens(self, img_t: torch.Tensor) -> torch.Tensor:
        """Forward teacher (frozen) → last-layer token sequence (B, N, D)."""
        with torch.no_grad():
            self.teacher.eval()
            out = self.teacher(img_t)["patch_tokens"]
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out

    def _compute_losses(
        self,
        s_bottleneck: torch.Tensor,   # (B, C_s, H, W)
        z_t: torch.Tensor,            # (B, N, D) teacher tokens
        prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (L_student, L_disc, z_s) where z_s = GL(s_bottleneck).
        """
        z_s = self.gl_proj(s_bottleneck)   # (B, N, D)

        # ── Auto-lambda: record initial scale once ────────────────────────────
        if prefix == "train" and not self._scales_set:
            with torch.no_grad():
                self._scale_gl.copy_(
                    F.mse_loss(z_s, z_t).clamp(min=1e-6)
                )
                if self.hparams.use_pca and self._qkv_cache is not None:
                    self._scale_pca.copy_(
                        self.pca_proj(s_bottleneck, self._qkv_cache).clamp(min=1e-6)
                    )
            self._scales_set.fill_(True)

        # GL loss
        L_gl = F.mse_loss(z_s, z_t.detach()) / self._scale_gl

        # PCA loss
        L_pca = z_s.new_zeros(())
        if self.hparams.use_pca and self._qkv_cache is not None:
            L_pca = self.pca_proj(s_bottleneck, self._qkv_cache) / self._scale_pca

        # Adversarial: student fools discriminator
        L_adv = F.binary_cross_entropy_with_logits(
            self.discriminator(z_s),
            torch.ones(z_s.shape[0], 1, device=z_s.device),
        )

        # Discriminator: real teacher → 1, fake projected student → 0
        L_disc = 0.5 * (
            F.binary_cross_entropy_with_logits(
                self.discriminator(z_t.detach()),
                torch.ones(z_t.shape[0], 1, device=z_t.device),
            )
            + F.binary_cross_entropy_with_logits(
                self.discriminator(z_s.detach()),
                torch.zeros(z_s.shape[0], 1, device=z_s.device),
            )
        )

        L_student = (
            L_gl
            + self.hparams.lambda_pca * L_pca
            + self.hparams.lambda_adv * L_adv
        )

        self.log_dict({
            f"{prefix}/gl_loss":   L_gl,
            f"{prefix}/pca_loss":  L_pca,
            f"{prefix}/adv_loss":  L_adv,
            f"{prefix}/disc_loss": L_disc,
        }, on_step=False, on_epoch=True, sync_dist=True)

        return L_student, L_disc, z_s

    # ── Lightning steps ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.student(x)

    def training_step(self, batch: dict, batch_idx: int) -> None:
        
        opt_s, opt_d = self.optimizers()
        sch_s, sch_d = self.lr_schedulers()

        img_s = batch["simulated"]
        img_t = batch["sentinel2"]

        # Teacher forward (hook fills self._qkv_cache if use_pca)
        z_t          = self._teacher_tokens(img_t)
        s_bottleneck = self.student(img_s)[-1]

        L_student, L_disc, _ = self._compute_losses(s_bottleneck, z_t, "train")
        
        # Student + projector update
        opt_s.zero_grad()
        self.manual_backward(L_student)
        opt_s.step()

        # Discriminator update (every disc_update_freq steps)
        self._step_count += 1
        if self._step_count % self.hparams.disc_update_freq == 0:
            opt_d.zero_grad()
            self.manual_backward(L_disc)
            opt_d.step()

        self.log("train_loss", L_student, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        if self.trainer.is_last_batch:
            sch_s.step()
            sch_d.step()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        img_s = batch["simulated"]
        img_t = batch["sentinel2"]

        z_t          = self._teacher_tokens(img_t)
        s_bottleneck = self.student(img_s)[-1]

        L_student, _, _ = self._compute_losses(s_bottleneck, z_t, "val")
        self.log("val_loss", L_student, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=img_s.shape[0])

    # ── Optimisers ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        lr      = self.hparams.lr
        wd      = self.hparams.weight_decay
        warmup  = self.hparams.warmup_epochs
        t_max   = max(1, self.trainer.max_epochs - warmup)

        student_params = [
            {"params": self.student.parameters(),    "lr": lr,        "name": "student"},
            {"params": self.gl_proj.parameters(),    "lr": lr * 10.,  "name": "gl_proj"},
        ]
        if self.hparams.use_pca and self.pca_proj is not None:
            student_params.append(
                {"params": self.pca_proj.parameters(), "lr": lr * 10., "name": "pca_proj"}
            )

        opt_s = torch.optim.AdamW(student_params, weight_decay=wd)
        # Discriminator: *0.5 (lower LR avoids discriminator dominating early)
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(), lr=lr * 0.5, weight_decay=wd
        )

        def _sched(opt):
            wu = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, end_factor=1.0, total_iters=warmup
            )
            co = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
            return torch.optim.lr_scheduler.SequentialLR(
                opt, [wu, co], milestones=[warmup]
            )

        return [opt_s, opt_d], [_sched(opt_s), _sched(opt_d)]