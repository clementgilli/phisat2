# ==========================================
# CONFIGURATION : Marine Area — Full Dataset
# ==========================================

MODEL       ?=

BASE_NAME   = marine_$(MODEL)_full
QUEUE       = gpu4_std
GPUS        = 1

TASK        = segmentation
DATASET     = marine
DATALOADER  = downstream_s2
ROOT_DIR    = /lustre/home/u10010021/phisat2/data

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
NUM_WORKERS = 8

_PRETRAIN = /lustre/home/u10010021/phisat2/runs/knowledge_distillation/ssl4eo/$(MODEL)/full_dataset/seed_42/checkpoints/best-v2.ckpt

# ─── eval / submit-eval ───────────────────────────────────────────────────────
ifneq ($(filter eval submit-eval, $(MAKECMDGOALS) $(TARGET)),)

JOB_NAME  = $(BASE_NAME)_eval
WALLTIME  = 02:00:00
CPUS      = 8
MEM       = 64gb

CKPT_PATH = /lustre/home/u10010021/phisat2/runs/segmentation/marine/$(MODEL)/full_dataset/seed_42/checkpoints/best-v3.ckpt

# ─── train / submit-train ─────────────────────────────────────────────────────
else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 12:00:00
CPUS       = 16
MEM        = 128gb

BATCH_SIZE = 32
LR         = 0.0001
WEIGHT_DECAY = 0.0001
EPOCHS     = 600

WEIGHTS    = $(_PRETRAIN)
SUBSET_CSV =

endif