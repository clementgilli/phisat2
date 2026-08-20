from __future__ import annotations

import torch
import torchmetrics

from phisat2.tasks import TaskSpec


MACRO_CLASS_MAPPINGS: dict[str, torch.Tensor] = {
    # 0:NoData, 1:Water, 2:Trees, 3:Grass, 4:FloodedVeg, 
    # 5:Crops, 6:Scrub, 7:Built, 8:Bare, 9:Snow/Ice, 10:Cloud
    
    # 0: Ignore (NoData, Cloud), 1: Vegetation (Trees, Grass, Crops, Scrub)
    # 2: Built Area (Built), 3: Bare/Ice (Bare, Snow/Ice), 4: Water (Water, FloodedVeg)
    "lulc": torch.tensor([0, 4, 1, 1, 4, 1, 1, 2, 3, 3, 0]),
    
    "clouds": torch.tensor([0, 1, 1, 0, 99])
}

# ─────────────────────────────────────────────────────────────────────────────
# Remapping wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _RemappedMetric(torchmetrics.Metric):

    full_state_update: bool = False

    def __init__(
        self,
        metric: torchmetrics.Metric,
        mapping: torch.Tensor,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.register_buffer("_mapping", mapping, persistent=False)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim > target.ndim:
            preds = preds.argmax(dim=1)
            
        self.metric.update(
            self._mapping[preds.long()],
            self._mapping[target.long()],
        )

    def compute(self) -> torch.Tensor:
        return self.metric.compute()

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
    
    metrics: dict[str, torchmetrics.Metric] = {}
    ignore_index = spec.ignore_index

    # ── Segmentation ─────────────────────────────────────────────────────────
    if spec.task == "segmentation":
        kw = dict(num_classes=spec.num_outputs, ignore_index=ignore_index)

        metrics[f"{prefix}_iou"] = torchmetrics.classification.MulticlassJaccardIndex(
            **kw, average="macro"
        )
        metrics[f"{prefix}_f1"] = torchmetrics.classification.MulticlassF1Score(
            **kw, average="macro"
        )
        metrics[f"{prefix}_acc"] = torchmetrics.classification.MulticlassAccuracy(
            **kw, average="micro"
        )

        mapping = MACRO_CLASS_MAPPINGS.get(spec.dataset)
        if mapping is not None:
            n_macro = int(mapping.max().item()) + 1
            macro_kw = dict(num_classes=n_macro, ignore_index=ignore_index)

            metrics[f"{prefix}_macro_iou"] = _RemappedMetric(
                torchmetrics.classification.MulticlassJaccardIndex(**macro_kw, average="macro"),
                mapping,
            )
            metrics[f"{prefix}_macro_f1"] = _RemappedMetric(
                torchmetrics.classification.MulticlassF1Score(**macro_kw, average="macro"),
                mapping,
            )
            metrics[f"{prefix}_macro_acc"] = _RemappedMetric(
                torchmetrics.classification.MulticlassAccuracy(**macro_kw, average="micro"),
                mapping,
            )

    # ── Classification ────────────────────────────────────────────────────────
    elif spec.task == "classification":
        kw = dict(num_classes=spec.num_outputs)
        metrics[f"{prefix}_f1"] = torchmetrics.classification.MulticlassF1Score(
            **kw, average="macro"
        )
        metrics[f"{prefix}_acc"] = torchmetrics.classification.MulticlassAccuracy(
            **kw, average="macro"
        )

    # ── Regression ────────────────────────────────────────────────────────────
    elif spec.task in {"pixel_regression", "global_regression"}:
        metrics[f"{prefix}_rmse"] = torchmetrics.regression.MeanSquaredError(squared=False)
        metrics[f"{prefix}_mae"]  = torchmetrics.regression.MeanAbsoluteError()

    return torchmetrics.MetricCollection(metrics)