# ==========================================
# CONFIGURATION : LULC — 500-shot
# ==========================================

BASE_NAME   = lulc_500shot
QUEUE       = gpu4_std
GPUS        = 1

TASK        = segmentation
DATASET     = lulc
MODEL       = phisatnet
DATALOADER  = downstream
ROOT_DIR    = /lustre/home/u10010021/phisat2/data

SEEDS       = 42 7 6
DEVICES     = 1
PRECISION   = bf16-mixed
NUM_WORKERS = 8

_PRETRAIN = /lustre/home/u10010021/phisat2/runs/pretrain_reconstruction/triplets/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

# ─── eval / submit-eval ───────────────────────────────────────────────────────
ifneq ($(filter eval submit-eval, $(MAKECMDGOALS) $(TARGET)),)

JOB_NAME  = $(BASE_NAME)_eval
WALLTIME  = 01:30:00
CPUS      = 8
MEM       = 48G

CKPT_PATH ?=

# ─── train / submit-train ─────────────────────────────────────────────────────
else

JOB_NAME  = $(BASE_NAME)_train
WALLTIME  = 24:00:00
CPUS      = 12
MEM       = 96G

BATCH_SIZE = 32
LR         = 0.0001
EPOCHS     = 200
PATIENCE   = 20

WEIGHTS    = $(_PRETRAIN)
SUBSET_CSV = /lustre/home/u10010021/phisat2/splits/lulc/lulc_train_500_global.csv

endif