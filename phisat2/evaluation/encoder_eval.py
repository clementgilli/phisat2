from __future__ import annotations

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from sklearn.manifold import TSNE
from pathlib import Path

LULC_MICRO = {
    0:  ("Tree Cover",         (34, 139, 34)),    
    1:  ("Shrubland",          (184, 134, 11)),   
    2:  ("Grassland",          (124, 252, 0)),   
    3:  ("Cropland",           (255, 215, 0)),   
    4:  ("Built-up",           (255, 0, 0)),    
    5:  ("Bare/Sparse Veg",    (210, 180, 140)),  
    6:  ("Snow and Ice",       (255, 255, 255)), 
    7:  ("Permanent Water",    (0, 0, 255)),     
    8:  ("Herbaceous Wetland", (0, 139, 139)),  
    9:  ("Mangroves",          (32, 178, 170)),  
    10: ("Moss and Lichen",    (240, 230, 140)),
}

LULC_MACRO = {
    0:  ("Vegetation", (34, 139, 34)),   
    1:  ("Vegetation", (34, 139, 34)),   
    2:  ("Vegetation", (34, 139, 34)),   
    3:  ("Vegetation", (34, 139, 34)),   
    10: ("Vegetation", (34, 139, 34)),   
    4:  ("Built-up",   (255, 0, 0)),      
    5:  ("Bare/Ice",   (210, 180, 140)),  
    6:  ("Bare/Ice",   (210, 180, 140)),  
    7:  ("Water",      (0, 0, 255)),      
    8:  ("Water",      (0, 0, 255)),      
    9:  ("Water",      (0, 0, 255)),      
}

def to_rgb_norm(color_tuple: tuple[int, int, int]) -> tuple[float, float, float]:
    return (color_tuple[0]/255.0, color_tuple[1]/255.0, color_tuple[2]/255.0)


class PretrainEvalModule(L.LightningModule):
    def __init__(self, full_model: nn.Module, max_samples: int = 3000):
        super().__init__()
        self.max_samples = max_samples
        self.feature_layers = ["enc_0", "enc_1", "enc_2", "bottleneck"]
        
        self.backbone = getattr(full_model, "encoder", getattr(full_model, "backbone", full_model))
        
        self.test_features = {layer: [] for layer in self.feature_layers}
        self.test_classes = {layer: [] for layer in self.feature_layers}

    def _to_named(self, features: dict | list | torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(features, dict): return features
        if isinstance(features, (list, tuple)):
            return {layer: features[i] for i, layer in enumerate(self.feature_layers)}
        return {"bottleneck": features} # Fallback

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        images = batch["image"]
        labels = batch["mask"]
        
        B = labels.shape[0]
        dominant_classes, _ = torch.mode(labels.view(B, -1), dim=1)
        
        feat_dict = self._to_named(self.backbone(images))
        
        for layer in self.feature_layers:
            if layer not in feat_dict: continue
            
            f = feat_dict[layer]
            f_1d = F.adaptive_avg_pool2d(f, 1).flatten(1)
            
            valid_mask = ~torch.isnan(f_1d).any(dim=1)
            
            if valid_mask.sum() > 0:
                self.test_features[layer].append(f_1d[valid_mask].cpu())
                self.test_classes[layer].append(dominant_classes[valid_mask].cpu())

    def on_test_epoch_end(self) -> None:
        out_dir = Path(self.trainer.default_root_dir)
        
        for layer in self.feature_layers:
            print(f"\n[INFO] Agrégation pour la couche : {layer}")
            
            if not self.test_features[layer]:
                print(f"[WARN] Aucune donnée valide pour {layer}, saut de l'étape.")
                continue

            X = torch.cat(self.test_features[layer], dim=0).numpy()
            Y = torch.cat(self.test_classes[layer], dim=0).numpy()
            
            if len(X) > self.max_samples:
                indices = np.random.choice(len(X), self.max_samples, replace=False)
                X, Y = X[indices], Y[indices]

            print(f"[INFO] Calcul t-SNE ({layer}) avec {len(X)} points...")
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
            emb = tsne.fit_transform(X)

            self._plot_micro(emb, Y, out_dir / f"tsne_{layer}_micro.png")
            self._plot_macro(emb, Y, out_dir / f"tsne_{layer}_macro.png")
            
            self.test_features[layer].clear()
            self.test_classes[layer].clear()

    def _plot_micro(self, emb, Y, save_path):
        plt.figure(figsize=(12, 10))
        legend_patches = []
        for class_id, (name, color) in LULC_MICRO.items():
            idx = (Y == class_id)
            if idx.sum() > 0:
                norm_color = to_rgb_norm(color)
                plt.scatter(emb[idx, 0], emb[idx, 1], c=[norm_color], label=name, s=20, alpha=0.8, edgecolors="none")
                legend_patches.append(mpatches.Patch(color=norm_color, label=name))
        
        plt.title("Latent Space Semantic Clustering (Micro)", fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_macro(self, emb, Y, save_path):
        plt.figure(figsize=(10, 8))
        macro_groups = {}
        for class_id, (macro_name, color) in LULC_MACRO.items():
            if macro_name not in macro_groups:
                macro_groups[macro_name] = {"color": to_rgb_norm(color), "indices": []}
            macro_groups[macro_name]["indices"].extend(np.where(Y == class_id)[0])

        legend_patches = []
        for macro_name, data in macro_groups.items():
            idx = np.array(data["indices"])
            if len(idx) > 0:
                plt.scatter(emb[idx, 0], emb[idx, 1], c=[data["color"]], label=macro_name, s=25, alpha=0.85, edgecolors="none")
                legend_patches.append(mpatches.Patch(color=data["color"], label=macro_name))

        plt.title("Latent Space Semantic Clustering (Macro)", fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()