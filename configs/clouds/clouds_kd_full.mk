# ==========================================
# CONFIGURATION : Clouds — Full Dataset
# ==========================================

MODEL       ?=

BASE_NAME   = clouds_$(MODEL)_full
QUEUE       = gpu4_std
GPUS        = 1

TASK        = segmentation
DATASET     = clouds
DATALOADER  = downstream_s2
ROOT_DIR    = /lustre/home/u10010021/phisat2/data

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
NUM_WORKERS = 8

_PRETRAIN = /lustre/home/u10010021/phisat2/runs/knowledge_distillation/ssl4eo/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt

# ─── eval / submit-eval ───────────────────────────────────────────────────────
ifneq ($(filter eval submit-eval, $(MAKECMDGOALS) $(TARGET)),)

JOB_NAME  = $(BASE_NAME)_eval
WALLTIME  = 02:00:00
CPUS      = 8
MEM       = 64gb

CKPT_PATH = /lustre/home/u10010021/phisat2/runs/segmentation/clouds/$(MODEL)/full_dataset/seed_42/checkpoints/best-v4.ckpt

# ─── train / submit-train ─────────────────────────────────────────────────────
else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 12:00:00
CPUS       = 16
MEM        = 128gb

BATCH_SIZE = 128
LR         = 0.0002
WEIGHT_DECAY = 0.001
EPOCHS     = 300
PATIENCE   = 20

WEIGHTS    = $(_PRETRAIN)
SUBSET_CSV =

endif