from __future__ import annotations

from dataclasses import dataclass


TASK_SEGMENTATION = "segmentation"
TASK_PIXEL_REGRESSION = "pixel_regression"
TASK_CLASSIFICATION = "classification"
TASK_GLOBAL_REGRESSION = "global_regression"

TASK_PRETRAIN_RECONSTRUCTION = "pretrain_reconstruction"
TASK_DISTILLATION_KD = "distillation_kd"

TASKS = {
    TASK_SEGMENTATION,
    TASK_PIXEL_REGRESSION,
    TASK_CLASSIFICATION,
    TASK_GLOBAL_REGRESSION,
    TASK_PRETRAIN_RECONSTRUCTION,
    TASK_DISTILLATION_KD,
}

SEGMENTATION_OUTPUTS = {
    "lulc": 11,
    "lc": 11,
    #"marine": 9,
    #"marine_area": 9,
    #"anomaly_detection": 9,
    #"burned": 4,
    #"burned_area": 4,
    #"clouds": 2,
    #"floods": 3,
    #"worldfloods": 3,
    #"fire": 3,
}

CLASSIFICATION_OUTPUTS = {
    "climate": 31,
}

GLOBAL_REGRESSION_OUTPUTS = {
    "geoloc": 4,
    "coords": 4,
}

PIXEL_REGRESSION_OUTPUTS = {
    "roads": 1,
    "building": 1,
}

PRETRAIN_RECONSTRUCTION_OUTPUTS = {
    "triplets": 8,
}

DISTILLATION_KD_OUTPUTS = {
    "triplets": 0,
}


@dataclass(frozen=True)
class TaskSpec:
    task: str
    dataset: str
    num_outputs: int
    target_key: str
    loss: str


def resolve_task_spec(task: str, dataset: str) -> TaskSpec:
    task = task.lower()
    dataset = dataset.lower()
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of: {', '.join(sorted(TASKS))}.")

    if task == TASK_SEGMENTATION:
        return TaskSpec(task, dataset, _lookup(dataset, SEGMENTATION_OUTPUTS), "mask", "cross_entropy")
    if task == TASK_CLASSIFICATION:
        return TaskSpec(task, dataset, _lookup(dataset, CLASSIFICATION_OUTPUTS), "label", "cross_entropy")
    if task == TASK_GLOBAL_REGRESSION:
        return TaskSpec(task, dataset, _lookup(dataset, GLOBAL_REGRESSION_OUTPUTS), "target", "mse")
    if task == TASK_PIXEL_REGRESSION:
        return TaskSpec(task, dataset, _lookup(dataset, PIXEL_REGRESSION_OUTPUTS), "target", "mse")
    
    
    if task == TASK_PRETRAIN_RECONSTRUCTION:
        return TaskSpec(task, dataset, _lookup(dataset, PRETRAIN_RECONSTRUCTION_OUTPUTS), "simulated", "mse")
        
    if task == TASK_DISTILLATION_KD:
        return TaskSpec(task, dataset, _lookup(dataset, DISTILLATION_KD_OUTPUTS), "none", "kd_loss")


def _lookup(dataset: str, outputs: dict[str, int]) -> int:
    try:
        return outputs[dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(outputs))
        raise ValueError(f"Dataset '{dataset}' is not valid for this task. Expected one of: {valid}.") from exc
