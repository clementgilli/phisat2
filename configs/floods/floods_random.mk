# ==========================================
# CONFIGURATION : Floods — Full Dataset
# ==========================================

BASE_NAME   = floods_random
QUEUE       = gpu4_std
GPUS        = 1

TASK        = segmentation
DATASET     = floods
MODEL       = phisatnet
DATALOADER  = downstream_s2
ROOT_DIR    = /lustre/home/u10010021/phisat2/data

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
NUM_WORKERS = 8

# ─── eval / submit-eval ───────────────────────────────────────────────────────
ifneq ($(filter eval submit-eval, $(MAKECMDGOALS) $(TARGET)),)

JOB_NAME  = $(BASE_NAME)_eval
WALLTIME  = 02:00:00
CPUS      = 8
MEM       = 64gb

CKPT_PATH ?=

# ─── train / submit-train ─────────────────────────────────────────────────────
else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 12:00:00
CPUS       = 16
MEM        = 128gb

EPOCHS     = 4
PATIENCE   = 10
BATCH_SIZE = 128
LR         = 0.0002
WEIGHT_DECAY = 0.00001

endif