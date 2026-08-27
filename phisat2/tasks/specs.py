from __future__ import annotations

from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# Task name constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_SEGMENTATION            = "segmentation"
TASK_PIXEL_REGRESSION        = "pixel_regression"
TASK_CLASSIFICATION          = "classification"
TASK_GLOBAL_REGRESSION       = "global_regression"
TASK_PRETRAIN_RECONSTRUCTION = "pretrain_reconstruction"
TASK_KNOWLEDGE_DISTILLATION  = "knowledge_distillation"
TASK_DOMAIN_ADAPTATION       = "domain_adaptation"
TASK_EVAL_DOMAIN_GAP         = "eval_domain_gap"
TASK_EVAL_ENCODER            = "eval_encoder"

TASKS: frozenset[str] = frozenset({
    TASK_SEGMENTATION,
    TASK_PIXEL_REGRESSION,
    TASK_CLASSIFICATION,
    TASK_GLOBAL_REGRESSION,
    TASK_PRETRAIN_RECONSTRUCTION,
    TASK_KNOWLEDGE_DISTILLATION,
    TASK_DOMAIN_ADAPTATION,
    TASK_EVAL_DOMAIN_GAP,
    TASK_EVAL_ENCODER,
})


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registries
# ─────────────────────────────────────────────────────────────────────────────
# Each downstream registry maps  dataset_name → num_outputs
# Segmentation also carries ignore_index: dataset → (num_classes, ignore_index)

_SEG: dict[str, tuple[int, int | None]] = {
    # dataset      classes  ignore   train    /  val    / test
    "lulc":    (11,  0),   # 50 080  /  5 552 /  6 544
    "floods":  (4,   0),   # 243 904 /  9 660 / 10 792
    "clouds":  (4,   99),  # 35 336  /  8 560 / 15 472
    "marine":  (9,   0),   #  7 936  /  1 280 /  3 584
    "methane": (2,   255), # 44 800  /  6 656 / 12 480
    "burned":  (7,   255), # 76 912  / 11 012 / 19 492
}

_PIX_REG: dict[str, int] = {
    "roads":    1,
    "building": 1,
}

_CLS: dict[str, int] = {
    "router":  5,
    "eurosat": 10,
}

_GLOBAL_REG: dict[str, int] = {}   # reserved for future tasks

# Non-downstream tasks
_PRETRAIN:  dict[str, int] = {"triplets": 8, "ssl4eo": 8}
_KD:        dict[str, int] = {"triplets": 0, "ssl4eo": 0}
_DA:        dict[str, int] = {"triplets": 0}
_EVAL_GAP:  dict[str, int] = {"triplets": 0}
_EVAL_ENC:  dict[str, int] = {"eurosat": 0}

# Flat reverse-lookup for guess_task_from_dataset (downstream tasks only)
_DOWNSTREAM_TASK: dict[str, str] = {
    **{d: TASK_SEGMENTATION    for d in _SEG},
    **{d: TASK_PIXEL_REGRESSION for d in _PIX_REG},
    **{d: TASK_CLASSIFICATION  for d in _CLS},
    **{d: TASK_GLOBAL_REGRESSION for d in _GLOBAL_REG},
}


# ─────────────────────────────────────────────────────────────────────────────
# TaskSpec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskSpec:
    task:         str
    dataset:      str
    num_outputs:  int
    target_key:   str
    loss:         str
    ignore_index: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(registry: dict, dataset: str, task: str):
    """Lookup dataset in a registry, raising a clear error on miss."""
    if dataset not in registry:
        valid = ", ".join(sorted(registry))
        raise ValueError(
            f"Dataset '{dataset}' is not registered for task '{task}'. "
            f"Expected one of: {valid}."
        )
    return registry[dataset]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def resolve_task_spec(task: str, dataset: str) -> TaskSpec:
    task    = task.lower()
    dataset = dataset.lower()

    if task not in TASKS:
        raise ValueError(
            f"Unknown task '{task}'. Expected one of: {', '.join(sorted(TASKS))}."
        )

    match task:
        case "segmentation":
            n, ign = _get(_SEG, dataset, task)
            return TaskSpec(task, dataset, n, "mask", "cross_entropy", ignore_index=ign)

        case "pixel_regression":
            return TaskSpec(task, dataset, _get(_PIX_REG, dataset, task), "target", "mse")

        case "classification":
            return TaskSpec(task, dataset, _get(_CLS, dataset, task), "label", "cross_entropy")

        case "global_regression":
            return TaskSpec(task, dataset, _get(_GLOBAL_REG, dataset, task), "target", "mse")

        case "pretrain_reconstruction":
            return TaskSpec(task, dataset, _get(_PRETRAIN, dataset, task), "simulated", "mse")

        case "knowledge_distillation":
            return TaskSpec(task, dataset, _get(_KD, dataset, task), "none", "kd_loss")

        case "domain_adaptation":
            return TaskSpec(task, dataset, _get(_DA, dataset, task), "none", "mse_multiscale")

        case "eval_domain_gap":
            return TaskSpec(task, dataset, _get(_EVAL_GAP, dataset, task), "none", "none")

        case "eval_encoder":
            return TaskSpec(task, dataset, _get(_EVAL_ENC, dataset, task), "mask", "none")

    raise AssertionError(f"Unhandled task '{task}'")


def guess_task_from_dataset(dataset: str) -> str:
    dataset = dataset.lower()
    if dataset not in _DOWNSTREAM_TASK:
        raise ValueError(
            f"Cannot infer task for unknown downstream dataset '{dataset}'. "
            f"Known datasets: {', '.join(sorted(_DOWNSTREAM_TASK))}."
        )
    return _DOWNSTREAM_TASK[dataset]