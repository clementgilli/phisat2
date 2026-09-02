from __future__ import annotations

import torch
import torchmetrics
from collections import OrderedDict

from phisat2.tasks import TaskSpec

# ─────────────────────────────────────────────────────────────────────────────
# Mappings Dictionaries (Juste les règles brutes)
# ─────────────────────────────────────────────────────────────────────────────

MACRO_CLASS_RULES: dict[str, dict[int, int]] = {
    # LULC Macro -> 1: Water, 2: Vegetation, 3: Built, 4: Bare/Ice
    "lulc": {
        1: 1,  # Water
        4: 1,  # Flooded Veg -> Water
        2: 2,  # Trees -> Veg
        3: 2,  # Grass -> Veg
        5: 2,  # Crops -> Veg
        6: 2,  # Scrub -> Veg
        7: 3,  # Built Area
        8: 4,  # Bare Ground
        9: 4   # Snow/Ice
    },
    
    # CLOUDS -> 0: Clear, 1: Cloud
    "clouds": {
        0: 0,  # Clear
        3: 0,  # Cloud Shadow -> Clear
        1: 1,  # Thick Cloud -> Cloud
        2: 1,  # Thin Cloud -> Cloud
    },
    
    # FLOODS -> 0: Land (Background), 1: Water (Foreground)
    "floods": {
        1: 0,  # Land
        2: 1,  # Water
    },
    
    # BURNED -> 0: Clear (Background), 1: Burned (Foreground)
    "burned": {
        0: 0,  # Clear
        5: 0,  # Water -> Clear/Non-Burned
        1: 1,  # Fresh Burn -> Burned
        2: 1,  # Old Burn -> Burned
    }
}

def _make_mapping(mapping_dict: dict[int, int], default: int, size: int = 256) -> torch.Tensor:
    """Crée un tenseur de mapping sécurisé."""
    t = torch.full((size,), default, dtype=torch.long)
    for k, v in mapping_dict.items():
        t[k] = v
    return t

# ─────────────────────────────────────────────────────────────────────────────
# Wrappers (Avec Double Mapping pour protéger les Prédictions)
# ─────────────────────────────────────────────────────────────────────────────

class _RemappedMetric(torchmetrics.Metric):
    full_state_update: bool = False

    def __init__(
        self,
        metric: torchmetrics.Metric,
        target_mapping: torch.Tensor,
        pred_mapping: torch.Tensor,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.register_buffer("_target_mapping", target_mapping, persistent=False)
        self.register_buffer("_pred_mapping", pred_mapping, persistent=False)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim > target.ndim:
            preds = preds.argmax(dim=1)
            
        self.metric.update(
            self._pred_mapping[preds.long()],
            self._target_mapping[target.long()],
        )

    def compute(self) -> torch.Tensor:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()
        super().reset()


class _ClassSpecificMetric(torchmetrics.Metric):
    full_state_update: bool = False

    def __init__(
        self,
        metric: torchmetrics.Metric,
        class_idx: int,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.class_idx = class_idx

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim > target.ndim:
            preds = preds.argmax(dim=1)
        self.metric.update(preds, target)

    def compute(self) -> torch.Tensor:
        result = self.metric.compute()
        if result.numel() > self.class_idx:
            return result[self.class_idx]
        return torch.tensor(float('nan'))

    def reset(self) -> None:
        self.metric.reset()
        super().reset()

# ─────────────────────────────────────────────────────────────────────────────
# Main factory
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics(
    spec: TaskSpec,
    prefix: str,
) -> torchmetrics.MetricCollection:
    
    metrics: dict[str, torchmetrics.Metric] = OrderedDict()
    
    ignore_index = spec.ignore_index 
    
    # ── Segmentation ─────────────────────────────────────────────────────────
    if spec.task == "segmentation":
        kw_base = dict(num_classes=spec.num_outputs, ignore_index=ignore_index)
        metrics[f"{prefix}_iou"] = torchmetrics.classification.MulticlassJaccardIndex(**kw_base, average="macro")
        metrics[f"{prefix}_f1"] = torchmetrics.classification.MulticlassF1Score(**kw_base, average="macro")

        rules = MACRO_CLASS_RULES.get(spec.dataset)
        
        if rules is not None:
            if spec.dataset == "lulc":
                SAFE_IGNORE = 5
                
                target_mapping = _make_mapping(rules, default=SAFE_IGNORE)
                
                pred_mapping = _make_mapping(rules, default=0)
                
                macro_kw = dict(num_classes=SAFE_IGNORE, ignore_index=SAFE_IGNORE)
                
                metrics[f"{prefix}_macro_iou"] = _RemappedMetric(
                    torchmetrics.classification.MulticlassJaccardIndex(**macro_kw, average="macro"), target_mapping, pred_mapping
                )
                metrics[f"{prefix}_macro_f1"] = _RemappedMetric(
                    torchmetrics.classification.MulticlassF1Score(**macro_kw, average="macro"), target_mapping, pred_mapping
                )
                
                iou_none = torchmetrics.classification.MulticlassJaccardIndex(**kw_base, average="none")
                for i in range(1, spec.num_outputs):
                    if i != ignore_index:
                        metrics[f"{prefix}_iou_class_{i}"] = _ClassSpecificMetric(iou_none, class_idx=i)

            elif spec.dataset in ["clouds", "floods", "burned"]:
                SAFE_IGNORE = 2 
                
                target_mapping = _make_mapping(rules, default=SAFE_IGNORE)
                pred_mapping = _make_mapping(rules, default=0) 
                
                bin_kw = dict(num_classes=SAFE_IGNORE, ignore_index=SAFE_IGNORE)
                
                base_iou_none = _RemappedMetric(
                    torchmetrics.classification.MulticlassJaccardIndex(**bin_kw, average="none"), target_mapping, pred_mapping
                )
                base_prec_none = _RemappedMetric(
                    torchmetrics.classification.MulticlassPrecision(**bin_kw, average="none"), target_mapping, pred_mapping
                )
                base_rec_none = _RemappedMetric(
                    torchmetrics.classification.MulticlassRecall(**bin_kw, average="none"), target_mapping, pred_mapping
                )
                
                target_name = "cloud" if spec.dataset == "clouds" else \
                              "flood" if spec.dataset == "floods" else "burned"
                
                metrics[f"{prefix}_{target_name}_iou"] = _ClassSpecificMetric(base_iou_none, class_idx=1)
                metrics[f"{prefix}_{target_name}_prec"] = _ClassSpecificMetric(base_prec_none, class_idx=1)
                metrics[f"{prefix}_{target_name}_rec"] = _ClassSpecificMetric(base_rec_none, class_idx=1)
                
                if spec.dataset == "clouds":
                    metrics[f"{prefix}_clear_iou"] = _ClassSpecificMetric(base_iou_none, class_idx=0)
                    
                metrics[f"{prefix}_macro_iou"] = _RemappedMetric(
                    torchmetrics.classification.MulticlassJaccardIndex(**bin_kw, average="macro"), target_mapping, pred_mapping
                )

    # ── Classification ────────────────────────────────────────────────────────
    elif spec.task == "classification":
        kw = dict(num_classes=spec.num_outputs)
        metrics[f"{prefix}_f1"] = torchmetrics.classification.MulticlassF1Score(**kw, average="macro")
        metrics[f"{prefix}_acc"] = torchmetrics.classification.MulticlassAccuracy(**kw, average="micro")

    # ── Regression ────────────────────────────────────────────────────────────
    elif spec.task in {"pixel_regression", "global_regression"}:
        metrics[f"{prefix}_rmse"] = torchmetrics.regression.MeanSquaredError(squared=False)
        metrics[f"{prefix}_mae"]  = torchmetrics.regression.MeanAbsoluteError()

    return torchmetrics.MetricCollection(metrics)