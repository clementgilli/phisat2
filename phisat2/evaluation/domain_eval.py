from __future__ import annotations

import os

import lightning as L
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torchmetrics.functional import jaccard_index

from phisat2.utils.visualization import mask_to_rgb


class DomainEvalModule(L.LightningModule):

    def __init__(
        self,
        teacher_encoder: nn.Module,   
        student_encoder: nn.Module,
        decoders: nn.ModuleDict,
    ) -> None:
        super().__init__()
        self.teacher  = teacher_encoder
        self.student  = student_encoder
        self.decoders = decoders

        self.feature_layers = ["enc_0", "enc_1", "enc_2", "bottleneck"]

        self.eval()
        self.requires_grad_(False)

        self.stored_sim    = {l: [] for l in self.feature_layers}   # teacher(sim)
        self.stored_before = {l: [] for l in self.feature_layers}   # teacher(real)
        self.stored_after  = {l: [] for l in self.feature_layers}   # student(real)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_named(feats: list | dict, layers: list[str]) -> dict[str, torch.Tensor]:
        if isinstance(feats, dict):
            return feats
        return {layers[i]: feats[i] for i in range(min(len(layers), len(feats)))}

    @staticmethod
    def _compute_pad(X_a: np.ndarray, X_b: np.ndarray) -> float:
        X = np.vstack([X_a, X_b])
        y = np.array([0] * len(X_a) + [1] * len(X_b))
        clf = make_pipeline(StandardScaler(), LinearSVC(C=0.01, dual=False, max_iter=10_000))
        acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1).mean()
        return float(2 * (1 - 2 * min(1.0 - acc, 0.5)))

    @staticmethod
    def _percentile_stretch(t: torch.Tensor, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
        flat = t.reshape(-1).float()
        v_lo = torch.quantile(flat, lo / 100.0)
        v_hi = torch.quantile(flat, hi / 100.0)
        return ((t.float() - v_lo) / (v_hi - v_lo + 1e-6)).clamp(0, 1).numpy()

    @staticmethod
    def _to_falsecolor(t: torch.Tensor, rgb_idx: tuple = (3, 2, 1)) -> np.ndarray:
        C = t.shape[0]
        idx = [c for c in rgb_idx if c < C]
        if len(idx) < 3:
            g = DomainEvalModule._percentile_stretch(t[0])
            return np.stack([g, g, g], axis=-1)
        return np.stack(
            [DomainEvalModule._percentile_stretch(t[c]) for c in idx], axis=-1
        )

    def _viz_dir(self, subfolder: str) -> str:
        d = os.path.join(self.trainer.default_root_dir, "visualizations", subfolder)
        os.makedirs(d, exist_ok=True)
        return d

    # ─────────────────────────────────────────────────────────────────────────
    # Visualisation
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_tsne_comparison(
        self,
        X_sim: np.ndarray,
        X_before: np.ndarray,
        X_after: np.ndarray,
        layer: str,
    ) -> None:
        
        from sklearn.manifold import TSNE

        MAX = 2000
        for arr, name in [(X_sim, "sim"), (X_before, "before"), (X_after, "after")]:
            if len(arr) > MAX:
                idx = np.random.choice(len(arr), MAX, replace=False)
                if name == "sim":    X_sim    = arr[idx]
                elif name == "before": X_before = arr[idx]
                else:                X_after  = arr[idx]

        n_sim, n_before, n_after = len(X_sim), len(X_before), len(X_after)
        X_all = np.vstack([X_sim, X_before, X_after])

        emb = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(X_all)
        e_sim    = emb[:n_sim]
        e_before = emb[n_sim: n_sim + n_before]
        e_after  = emb[n_sim + n_before:]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"t-SNE — {layer}", fontsize=14, fontweight="bold")

        kw = dict(s=12, alpha=0.85, edgecolors="none")
        color_sim    = "mediumblue"
        color_before = "crimson"
        color_after  = "crimson"

        axes[0].set_title("Before DA")
        axes[0].scatter(e_sim[:,0],    e_sim[:,1],    color=color_sim,    label="teacher(sim)",   **kw)
        axes[0].scatter(e_before[:,0], e_before[:,1], color=color_before, label="teacher(real)",  **kw)
        axes[0].legend(markerscale=2, fontsize=10)
        axes[0].axis("off")

        axes[1].set_title("After DA")
        axes[1].scatter(e_sim[:,0],   e_sim[:,1],   color=color_sim,   label="teacher(sim)",   **kw)
        axes[1].scatter(e_after[:,0], e_after[:,1], color=color_after, label="student(real)",  **kw)
        axes[1].legend(markerscale=2, fontsize=10)
        axes[1].axis("off")

        plt.tight_layout()
        
        save_dir = self._viz_dir("tsne")
        plt.savefig(os.path.join(save_dir, f"{layer}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _visualize_seg(
        self,
        img_sim: torch.Tensor,
        img_real: torch.Tensor,
        logits_sim: torch.Tensor,
        logits_before: torch.Tensor,
        logits_after: torch.Tensor,
        task_name: str,
        batch_idx: int,
        max_samples: int = 5,
    ) -> None:
        
        def argmax(t: torch.Tensor) -> np.ndarray:
            return t.argmax(dim=1).detach().cpu().numpy() if t.ndim == 4 else t.detach().cpu().numpy()

        preds_sim    = argmax(logits_sim)
        preds_before = argmax(logits_before)
        preds_after  = argmax(logits_after)
        
        n = min(max_samples, img_sim.shape[0])
        
        sampled_indices = np.random.choice(img_sim.shape[0], n, replace=False)

        fig, axes = plt.subplots(n, 5, figsize=(25, 4 * n), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Consistency Segmentation — {task_name} — Batch {batch_idx}", fontsize=14)
        for ax, title in zip(
            axes[0],
            ["Image SIM", "Pred SIM", "Image REAL", "Pred REAL (before DA)", "Pred REAL (after DA)"],
        ):
            ax.set_title(title, fontsize=11)

        meta = None
        for row, idx in enumerate(sampled_indices):
            axes[row, 0].imshow(self._to_falsecolor(img_sim[idx].cpu()))
            axes[row, 2].imshow(self._to_falsecolor(img_real[idx].cpu()))
            for ax_col, pred_np in [(1, preds_sim), (3, preds_before), (4, preds_after)]:
                rgb, meta = mask_to_rgb(pred_np[idx], task_name)
                axes[row, ax_col].imshow(rgb)
            for ax in axes[row]:
                ax.axis("off")

        if meta:
            patches = [
                mpatches.Patch(color=np.array(c) / 255.0, label=n_)
                for _, (n_, c) in meta.items()
            ]
            fig.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.04),
                       ncol=len(meta), fontsize=10)

        save_dir = self._viz_dir(f"seg_{task_name}")
        plt.savefig(
            os.path.join(save_dir, f"batch_{batch_idx}.png"),
            dpi=120, bbox_inches="tight",
        )
        plt.close(fig)

    def _visualize_reg(
        self,
        img_sim: torch.Tensor,
        img_real: torch.Tensor,
        preds_sim: torch.Tensor,
        preds_before: torch.Tensor,
        preds_after: torch.Tensor,
        task_name: str,
        batch_idx: int,
        max_samples: int = 5,
    ) -> None:
        def squeeze(t: torch.Tensor) -> np.ndarray:
            t = t if t.ndim == 3 else (t.squeeze(1) if t.shape[1] == 1 else t[:, 0])
            return t.detach().cpu().float().numpy()

        ps = squeeze(preds_sim)
        pb = squeeze(preds_before)
        pa = squeeze(preds_after)
        vmin = min(np.percentile(ps, 2), np.percentile(pb, 2), np.percentile(pa, 2))
        vmax = max(np.percentile(ps, 98), np.percentile(pb, 98), np.percentile(pa, 98))
        
        n = min(max_samples, img_sim.shape[0])
        
        sampled_indices = np.random.choice(img_sim.shape[0], n, replace=False)

        fig, axes = plt.subplots(n, 5, figsize=(25, 4 * n), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Consistency Regression — {task_name} — Batch {batch_idx}", fontsize=14)
        for ax, title in zip(
            axes[0],
            ["Image SIM", "Pred SIM", "Image REAL", "Pred REAL (before DA)", "Pred REAL (after DA)"],
        ):
            ax.set_title(title, fontsize=11)

        im_ref = None
        for row, idx in enumerate(sampled_indices):
            axes[row, 0].imshow(self._to_falsecolor(img_sim[idx].cpu()))
            axes[row, 2].imshow(self._to_falsecolor(img_real[idx].cpu()))
            for ax_col, pred in [(1, ps), (3, pb), (4, pa)]:
                im_ref = axes[row, ax_col].imshow(pred[idx], cmap="viridis", vmin=vmin, vmax=vmax)
            for ax in axes[row]:
                ax.axis("off")

        if im_ref is not None:
            fig.colorbar(im_ref, ax=axes[:, [1, 3, 4]], shrink=0.7, pad=0.02)

        save_dir = self._viz_dir(f"reg_{task_name}")
        plt.savefig(
            os.path.join(save_dir, f"batch_{batch_idx}.png"),
            dpi=120, bbox_inches="tight",
        )
        plt.close(fig)
        
    def _visualize_cls(
        self,
        img_sim: torch.Tensor,
        img_real: torch.Tensor,
        logits_sim: torch.Tensor,
        logits_before: torch.Tensor,
        logits_after: torch.Tensor,
        task_name: str,
        batch_idx: int,
        max_samples: int = 5,
    ) -> None:
        
        def argmax(t: torch.Tensor) -> np.ndarray:
            return t.argmax(dim=1).detach().cpu().numpy() if t.ndim == 2 else t.detach().cpu().numpy()

        preds_sim    = argmax(logits_sim).astype(int)
        preds_before = argmax(logits_before).astype(int)
        preds_after  = argmax(logits_after).astype(int)
        
        n = min(max_samples, img_sim.shape[0])
        
        sampled_indices = np.random.choice(img_sim.shape[0], n, replace=False)

        fig, axes = plt.subplots(n, 5, figsize=(25, 4 * n), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Consistency Classification — {task_name} — Batch {batch_idx}", fontsize=16, fontweight='bold')
        
        col_titles = ["Image SIM", "Pred SIM", "Image REAL", "Pred REAL (before DA)", "Pred REAL (after DA)"]
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=14)

        current_meta = None
        
        for row, idx in enumerate(sampled_indices):
            axes[row, 0].imshow(self._to_falsecolor(img_sim[idx].cpu()))            
            axes[row, 2].imshow(self._to_falsecolor(img_real[idx].cpu()))
            
            for ax_col, pred_array in [(1, preds_sim), (3, preds_before), (4, preds_after)]:
                pred_idx = pred_array[idx]
                
                dummy_mask = np.full((128, 128), pred_idx, dtype=np.uint8)
                rgb_img, current_meta = mask_to_rgb(dummy_mask, task_name)
                
                axes[row, ax_col].imshow(rgb_img)
                
                pred_name, pred_rgb = current_meta.get(pred_idx, ("Unknown", (0, 0, 0)))
                
                luminance = 0.299 * pred_rgb[0] + 0.587 * pred_rgb[1] + 0.114 * pred_rgb[2]
                text_color = "black" if luminance > 150 else "white"
                
                axes[row, ax_col].text(64, 64, pred_name, ha="center", va="center", 
                                       color=text_color, fontsize=18, fontweight="bold")
                
            for ax in axes[row]:
                ax.axis("off")

        if current_meta:
            patches = [
                mpatches.Patch(color=np.array(color) / 255.0, label=name)
                for _, (name, color) in current_meta.items()
            ]
            fig.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.04),
                       ncol=len(current_meta), fontsize=14)

        save_dir = self._viz_dir(f"cls_{task_name}")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"batch_{batch_idx}.png")
        
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[Viz] saved → {save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Test step
    # ─────────────────────────────────────────────────────────────────────────

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        img_sim  = batch["simulated"]
        img_real = batch["real"]

        feat_sim    = self._to_named(self.teacher(img_sim),  self.feature_layers)
        feat_before = self._to_named(self.teacher(img_real), self.feature_layers)
        feat_after  = self._to_named(self.student(img_real), self.feature_layers)

        for layer in self.feature_layers:
            if layer not in feat_sim:
                continue

            f_sim    = F.adaptive_avg_pool2d(feat_sim[layer],    1).flatten(1)
            f_before = F.adaptive_avg_pool2d(feat_before[layer], 1).flatten(1)
            f_after  = F.adaptive_avg_pool2d(feat_after[layer],  1).flatten(1)

            self.log(f"before/cosine_{layer}", F.cosine_similarity(f_sim, f_before).mean(), on_step=False, on_epoch=True)
            self.log(f"after/cosine_{layer}",  F.cosine_similarity(f_sim, f_after).mean(),  on_step=False, on_epoch=True)

            self.stored_sim[layer].append(f_sim.cpu().numpy())
            self.stored_before[layer].append(f_before.cpu().numpy())
            self.stored_after[layer].append(f_after.cpu().numpy())

        # ── Consistency downstream ─────────────────────────────────────────────
        do_plot = (batch_idx % 5 == 0)
        for task_name, decoder in self.decoders.items():
            feat_list_sim    = [feat_sim[l]    for l in self.feature_layers if l in feat_sim]
            feat_list_before = [feat_before[l] for l in self.feature_layers if l in feat_before]
            feat_list_after  = [feat_after[l]  for l in self.feature_layers if l in feat_after]

            logits_sim    = decoder(feat_list_sim)
            logits_before = decoder(feat_list_before)
            logits_after  = decoder(feat_list_after)

            is_spatial   = logits_sim.ndim == 4
            n_classes    = logits_sim.shape[1]

            if is_spatial and n_classes > 1:
                # ── Segmentation ─────────────────────────────────────────────
                for prefix, logits_real in [("before", logits_before), ("after", logits_after)]:
                    preds_sim_hard  = logits_sim.argmax(1)
                    preds_real_hard = logits_real.argmax(1)

                    miou = jaccard_index(
                        preds_real_hard, preds_sim_hard,
                        task="multiclass", num_classes=n_classes, average="macro",
                    )
                    p_sim  = F.softmax(logits_sim, 1)
                    kl = torch.sum(
                        p_sim * (F.log_softmax(logits_sim, 1) - F.log_softmax(logits_real, 1)),
                        dim=1,
                    ).mean()
                    self.log(f"{prefix}/consistency_{task_name}_miou", miou, on_step=False, on_epoch=True)
                    self.log(f"{prefix}/consistency_{task_name}_kl",   kl,   on_step=False, on_epoch=True)

                if do_plot:
                    self._visualize_seg(
                        img_sim, img_real,
                        logits_sim, logits_before, logits_after,
                        task_name, batch_idx,
                    )

            elif not is_spatial and n_classes > 1:
                # ── Global Classification ────────────────────────────────────
                for prefix, logits_real in [("before", logits_before), ("after", logits_after)]:
                    acc = (logits_real.argmax(1) == logits_sim.argmax(1)).float().mean()
                    kl = torch.sum(
                        F.softmax(logits_sim, 1) * (
                            F.log_softmax(logits_sim, 1) - F.log_softmax(logits_real, 1)
                        ),
                        dim=1,
                    ).mean()
                    self.log(f"{prefix}/consistency_{task_name}_acc", acc, on_step=False, on_epoch=True)
                    self.log(f"{prefix}/consistency_{task_name}_kl",  kl,  on_step=False, on_epoch=True)
                
                if do_plot:
                    self._visualize_cls(
                        img_sim, img_real,
                        logits_sim, logits_before, logits_after,
                        task_name, batch_idx,
                    )

            else:
                # ── Regression ────────────────────────────────────────────────
                for prefix, logits_real in [("before", logits_before), ("after", logits_after)]:
                    mse = F.mse_loss(logits_real, logits_sim)
                    self.log(f"{prefix}/consistency_{task_name}_mse", mse, on_step=False, on_epoch=True)

                if do_plot and is_spatial:
                    self._visualize_reg(
                        img_sim, img_real,
                        logits_sim, logits_before, logits_after,
                        task_name, batch_idx,
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # End epoch : PAD + t-SNE
    # ─────────────────────────────────────────────────────────────────────────

    def on_test_epoch_end(self) -> None:
        print("\n" + "=" * 60)
        print("  PAD SCORES — Before vs After DA")
        print("=" * 60)
        print(f"  {'Layer':<12} {'PAD Before':>12} {'PAD After':>12}  {'Diff':>10}")
        print("-" * 60)

        for layer in self.feature_layers:
            if not self.stored_sim[layer]:
                continue

            X_sim    = np.concatenate(self.stored_sim[layer])
            X_before = np.concatenate(self.stored_before[layer])
            X_after  = np.concatenate(self.stored_after[layer])

            pad_before = self._compute_pad(X_sim, X_before)
            pad_after  = self._compute_pad(X_sim, X_after)
            delta      = pad_after - pad_before

            self.log(f"before/pad_{layer}", pad_before)
            self.log(f"after/pad_{layer}",  pad_after)

            sign = "↓" if delta < 0 else "↑"
            print(f"  {layer:<12} {pad_before:>12.4f} {pad_after:>12.4f}  {sign} {abs(delta):.4f}")

            self._plot_tsne_comparison(X_sim, X_before, X_after, layer)

        print("=" * 60 + "\n")