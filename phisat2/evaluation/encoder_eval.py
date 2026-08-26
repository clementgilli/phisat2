from __future__ import annotations

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
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
    0: ("Vegetation", (34, 139, 34)),
    1: ("Vegetation", (34, 139, 34)),
    2: ("Vegetation", (34, 139, 34)),
    3: ("Vegetation", (34, 139, 34)),
    10: ("Vegetation", (34, 139, 34)),
    4: ("Built-up", (255, 0, 0)),
    5: ("Bare/Ice", (210, 180, 140)),
    6: ("Bare/Ice", (210, 180, 140)),
    7: ("Water", (0, 0, 255)),
    8: ("Water", (0, 0, 255)),
    9: ("Water", (0, 0, 255)),
}

LULC_MICRO_TO_MACRO_MAP = {
    0: 0, 1: 0, 2: 0, 3: 0, 10: 0,
    4: 1,
    5: 2, 6: 2,
    7: 3, 8: 3, 9: 3
}

EVENTS_ROUTER = {
    0: ("Safe",   (34, 139, 34)),
    1: ("Fire",   (255, 69, 0)),     
    2: ("Burnt",  (105, 105, 105)), 
    3: ("Water",  (30, 144, 255)), 
    4: ("Clouds", (220, 220, 220)),
}

EUROSAT_CLASSES = {
    0: ("AnnualCrop",           (230, 230, 0)),   
    1: ("Forest",               (34, 139, 34)),   
    2: ("HerbaceousVegetation", (144, 238, 144)), 
    3: ("Highway",              (105, 105, 105)), 
    4: ("Industrial",           (220, 20, 60)),     
    5: ("Pasture",              (173, 255, 47)),   
    6: ("PermanentCrop",        (218, 165, 32)),  
    7: ("Residential",          (139, 69, 19)),   
    8: ("River",                (0, 191, 255)),   
    9: ("SeaLake",              (0, 0, 128)),     
}

def to_rgb_norm(color_tuple: tuple[int, int, int]) -> tuple[float, float, float]:
    return (color_tuple[0]/255.0, color_tuple[1]/255.0, color_tuple[2]/255.0)


class PretrainEvalModule(L.LightningModule):
    def __init__(
        self, 
        full_model: nn.Module, 
        spec: TaskSpec,
        max_samples: int = 3000
    ):
        super().__init__()
        self.max_samples = max_samples
        self.spec = spec
        self.feature_layers = ["enc_0", "enc_1", "enc_2", "bottleneck"]
        
        self.backbone = getattr(full_model, "encoder", getattr(full_model, "backbone", full_model))
        
        self.test_features = {layer: [] for layer in self.feature_layers}
        self.test_classes = {layer: [] for layer in self.feature_layers}

        if self.spec.dataset == "lulc":
            self.has_macro = True
            self.class_dict_micro = LULC_MICRO
            self.class_dict_macro = LULC_MACRO
            self.micro_to_macro_map = LULC_MICRO_TO_MACRO_MAP
        elif self.spec.dataset == "router":
            self.has_macro = False
            self.class_dict_micro = EVENTS_ROUTER
        elif self.spec.dataset == "eurosat":
            self.has_macro = False
            self.class_dict_micro = EUROSAT_CLASSES
        else:
            raise ValueError(f"Dataset not supported : {self.spec.dataset}")

    def _to_named(self, features: dict | list | torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(features, dict): return features
        if isinstance(features, (list, tuple)):
            return {layer: features[i] for i, layer in enumerate(self.feature_layers)}
        return {"bottleneck": features}
                
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        images = batch["sentinel2_phisat2"]
        labels = batch[self.spec.target_key]
        
        B = labels.shape[0]
        
        if labels.ndim > 1:
            target_classes, _ = torch.mode(labels.view(B, -1), dim=1)
        else:
            target_classes = labels
        
        feat_dict = self._to_named(self.backbone(images))
        
        for layer in self.feature_layers:
            if layer not in feat_dict: continue
            
            f = feat_dict[layer]
            f_1d = F.adaptive_avg_pool2d(f, 1).flatten(1)
            
            valid_mask = ~torch.isnan(f_1d).any(dim=1)
            
            if valid_mask.sum() > 0:
                self.test_features[layer].append(f_1d[valid_mask].cpu())
                self.test_classes[layer].append(target_classes[valid_mask].cpu())

    def _evaluate_few_shot_1nn(self, X: np.ndarray, Y: np.ndarray, n_shots: int = 1, runs: int = 10) -> float:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        
        accs = []
        unique_classes = np.unique(Y)
        
        for _ in range(runs):
            support_X, support_Y = [], []
            query_X, query_Y = [], []
            
            for c in unique_classes:
                idx = np.where(Y == c)[0]
                if len(idx) <= n_shots:
                    continue 
                
                np.random.shuffle(idx)
                support_X.append(X[idx[:n_shots]])
                support_Y.append(Y[idx[:n_shots]])
                query_X.append(X[idx[n_shots:]])
                query_Y.append(Y[idx[n_shots:]])
                
            if not support_X:
                continue
                
            X_sup, Y_sup = np.concatenate(support_X), np.concatenate(support_Y)
            X_que, Y_que = np.concatenate(query_X), np.concatenate(query_Y)
            
            knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
            knn.fit(X_sup, Y_sup)
            preds = knn.predict(X_que)
            
            accs.append(accuracy_score(Y_que, preds))
            
        return float(np.mean(accs)) if accs else 0.0

    def on_test_epoch_end(self) -> None:
        out_dir = Path(self.trainer.default_root_dir)
        
        for layer in self.feature_layers:
            if not self.test_features[layer]:
                continue

            X = torch.cat(self.test_features[layer], dim=0).numpy()
            Y_micro = torch.cat(self.test_classes[layer], dim=0).numpy()
            
            acc_micro_1shot = self._evaluate_few_shot_1nn(X, Y_micro, n_shots=1, runs=200)
            acc_micro_5shot = self._evaluate_few_shot_1nn(X, Y_micro, n_shots=5, runs=200)
            
            self.log(f"few_shot/{self.spec.dataset}_1_shot_{layer}", acc_micro_1shot)
            self.log(f"few_shot/{self.spec.dataset}_5_shot_{layer}", acc_micro_5shot)
            
            if self.has_macro:
                Y_macro = np.array([self.micro_to_macro_map.get(y, -1) for y in Y_micro])
                valid_mask = Y_macro != -1
                X_valid, Y_macro_valid = X[valid_mask], Y_macro[valid_mask]
                
                if len(X_valid) > 0:
                    acc_macro_1shot = self._evaluate_few_shot_1nn(X_valid, Y_macro_valid, n_shots=1, runs=200)
                    acc_macro_5shot = self._evaluate_few_shot_1nn(X_valid, Y_macro_valid, n_shots=5, runs=200)
                    self.log(f"few_shot/{self.spec.dataset}_macro_1_shot_{layer}", acc_macro_1shot)
                    self.log(f"few_shot/{self.spec.dataset}_macro_5_shot_{layer}", acc_macro_5shot)

            indices = np.random.choice(len(X), min(len(X), self.max_samples), replace=False)
            X_tsne, Y_tsne = X[indices], Y_micro[indices] 

            tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
            emb = tsne.fit_transform(X_tsne)

            self._plot_clusters(emb, Y_tsne, self.class_dict_micro, out_dir / f"tsne_{layer}_{self.spec.dataset}.pdf")
            
            if self.has_macro:
                self._plot_clusters(emb, Y_tsne, self.class_dict_macro, out_dir / f"tsne_{layer}_lulc_macro.pdf")

            self.test_features[layer].clear()
            self.test_classes[layer].clear()

    def _plot_clusters(self, emb, Y, class_dict, save_path):
        plt.figure(figsize=(12, 10))
        legend_patches = []
        
        name_groups = {}
        for class_id, (name, color) in class_dict.items():
            if name not in name_groups:
                name_groups[name] = {"color": to_rgb_norm(color), "indices": []}
            name_groups[name]["indices"].extend(np.where(Y == class_id)[0])

        for name, data in name_groups.items():
            idx = np.array(data["indices"])
            if len(idx) > 0:
                plt.scatter(emb[idx, 0], emb[idx, 1], c=[data["color"]], label=name, s=20, alpha=0.8, edgecolors="none")
                legend_patches.append(mpatches.Patch(color=data["color"], label=name))
        
        plt.title(f"Latent Space Semantic Clustering", fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close()