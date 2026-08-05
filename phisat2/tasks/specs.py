from __future__ import annotations

from dataclasses import dataclass

TASK_SEGMENTATION = "segmentation"
TASK_PIXEL_REGRESSION = "pixel_regression"
TASK_CLASSIFICATION = "classification"
TASK_GLOBAL_REGRESSION = "global_regression"

TASK_PRETRAIN_RECONSTRUCTION = "pretrain_reconstruction"
TASK_EVAL_ENCODER = "eval_encoder"

TASK_KNOWLEDGE_DISTILLATION = "knowledge_distillation"

TASK_DOMAIN_ADAPTATION = "domain_adaptation"
TASK_EVAL_DOMAIN_GAP = "eval_domain_gap" 

TASKS = {
    TASK_SEGMENTATION,
    TASK_PIXEL_REGRESSION,
    TASK_CLASSIFICATION,
    TASK_GLOBAL_REGRESSION,
    TASK_PRETRAIN_RECONSTRUCTION,
    TASK_KNOWLEDGE_DISTILLATION,
    TASK_DOMAIN_ADAPTATION,
    TASK_EVAL_DOMAIN_GAP,
    TASK_EVAL_ENCODER,
}

SEGMENTATION_OUTPUTS = {
    "lulc": 11,
    "lc": 11,
    #"marine": 9,
    #"marine_area": 9,
    #"anomaly_detection": 9,
    "burned": 4,
    #"burned_area": 4,
    "clouds": 4,
    "floods": 3,
    #"worldfloods": 3,
}

PIXEL_REGRESSION_OUTPUTS = {
    "roads": 1,
    "building": 1,
}

GLOBAL_REGRESSION_OUTPUTS = {
}

CLASSIFICATION_OUTPUTS = {
    "router": 5,
}

PRETRAIN_RECONSTRUCTION_OUTPUTS = {
    "triplets": 8,
}

KNOWLEDGE_DISTILLATION_OUTPUTS = {
    "triplets": 0,
}

DOMAIN_ADAPTATION_OUTPUTS = {
    "triplets": 0,
}

EVAL_DOMAIN_GAP_OUTPUTS = {
    "triplets": 0,
}

EVAL_ENCODER_OUTPUTS = {
    "lulc": 0,
    "router": 0,
    "eurosat": 0,
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
        
    if task == TASK_KNOWLEDGE_DISTILLATION:
        return TaskSpec(task, dataset, _lookup(dataset, KNOWLEDGE_DISTILLATION_OUTPUTS), "none", "kd_loss")
    
    if task == TASK_DOMAIN_ADAPTATION:
        return TaskSpec(task, dataset, _lookup(dataset, DOMAIN_ADAPTATION_OUTPUTS), "none", "mse_multiscale")
        
    if task == TASK_EVAL_DOMAIN_GAP:
        return TaskSpec(task, dataset, _lookup(dataset, EVAL_DOMAIN_GAP_OUTPUTS), "none", "none")
    
    if task == TASK_EVAL_ENCODER:
        return TaskSpec(task, dataset, _lookup(dataset, EVAL_ENCODER_OUTPUTS), "mask", "none")

def _lookup(dataset: str, outputs: dict[str, int]) -> int:
    try:
        return outputs[dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(outputs))
        raise ValueError(f"Dataset '{dataset}' is not valid for this task. Expected one of: {valid}.") from exc

def guess_task_from_dataset(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset in SEGMENTATION_OUTPUTS: return TASK_SEGMENTATION
    if dataset in PIXEL_REGRESSION_OUTPUTS: return TASK_PIXEL_REGRESSION
    if dataset in CLASSIFICATION_OUTPUTS: return TASK_CLASSIFICATION
    if dataset in GLOBAL_REGRESSION_OUTPUTS: return TASK_GLOBAL_REGRESSION
    
    raise ValueError(f"Cannot infer task type for unknown downstream dataset '{dataset}'.")