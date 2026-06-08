import torchmetrics
from phisat2.tasks import TaskSpec

def build_metrics(
    spec: TaskSpec,
    prefix: str,
    ignore_index: int | None = None,
) -> torchmetrics.MetricCollection:
    metrics = {}

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

    elif spec.task == "classification":
        kw = dict(num_classes=spec.num_outputs)
        metrics[f"{prefix}_f1"] = torchmetrics.classification.MulticlassF1Score(
            **kw, average="macro"
        )
        metrics[f"{prefix}_acc"] = torchmetrics.classification.MulticlassAccuracy(
            **kw, average="macro"  
        )

    elif spec.task in ["pixel_regression", "global_regression"]:
        metrics[f"{prefix}_rmse"] = torchmetrics.regression.MeanSquaredError(squared=False)
        metrics[f"{prefix}_mae"]  = torchmetrics.regression.MeanAbsoluteError()
        metrics[f"{prefix}_r2"]   = torchmetrics.regression.R2Score()

    return torchmetrics.MetricCollection(metrics)