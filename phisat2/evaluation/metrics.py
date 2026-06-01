import torchmetrics
from phisat2.tasks import TaskSpec

def build_metrics(spec: TaskSpec, prefix: str) -> torchmetrics.MetricCollection:
    
    metrics = {}
    
    if spec.task == "segmentation":
        # 1. IoU (Jaccard)
        metrics[f"{prefix}_iou"] = torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=spec.num_outputs, 
            average="macro"
        )
        # 2. F1 Score (Dice)
        metrics[f"{prefix}_f1"] = torchmetrics.classification.MulticlassF1Score(
            num_classes=spec.num_outputs, 
            average="macro"
        )
        # 3. Pixel Accuracy
        metrics[f"{prefix}_acc"] = torchmetrics.classification.MulticlassAccuracy(
            num_classes=spec.num_outputs,
            average="micro"
        )
        
    elif spec.task == "classification":
        # 1. F1 Score 
        metrics[f"{prefix}_f1"] = torchmetrics.classification.MulticlassF1Score(
            num_classes=spec.num_outputs,
            average="macro"
        )
        # 2. Accuracy
        metrics[f"{prefix}_acc"] = torchmetrics.classification.MulticlassAccuracy(
            num_classes=spec.num_outputs,
            average="micro"
        )
        
    elif spec.task in ["pixel_regression", "global_regression"]:
        metrics[f"{prefix}_rmse"] = torchmetrics.regression.MeanSquaredError(squared=False)
        metrics[f"{prefix}_mae"] = torchmetrics.regression.MeanAbsoluteError()

    return torchmetrics.MetricCollection(metrics)