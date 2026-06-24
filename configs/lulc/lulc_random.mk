# ==========================================
# CONFIGURATION : LULC — Full Dataset
# ==========================================

BASE_NAME   = lulc_full
QUEUE       = gpu4_std
GPUS        = 1

TASK        = segmentation
DATASET     = lulc
MODEL       = phisatnet
DATALOADER  = downstream
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
MEM       = 64G

CKPT_PATH ?=

# ─── train / submit-train ─────────────────────────────────────────────────────
else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 12:00:00
CPUS       = 16
MEM        = 128G

BATCH_SIZE = 128
LR         = 0.001
EPOCHS     = 100

WEIGHTS    = $(_PRETRAIN)
SUBSET_CSV =

endif