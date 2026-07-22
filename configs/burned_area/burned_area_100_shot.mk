# ==========================================
# CONFIGURATION : BURNED_AREA — 50-shot
# ==========================================

BASE_NAME   = burned_50shot
QUEUE       = gpu4_std
GPUS        = 1

TASK        = segmentation
DATASET     = burned
MODEL       = phisatnet
DATALOADER  = downstream
ROOT_DIR    = /lustre/home/u10010021/phisat2/data

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
NUM_WORKERS = 4

_PRETRAIN = /lustre/home/u10010021/phisat2/runs/pretrain_reconstruction/triplets/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

# ─── eval / submit-eval ───────────────────────────────────────────────────────
ifneq ($(filter eval submit-eval, $(MAKECMDGOALS) $(TARGET)),)

JOB_NAME  = $(BASE_NAME)_eval
WALLTIME  = 01:00:00
CPUS      = 4
MEM       = 32G

CKPT_PATH ?=

# ─── train / submit-train ─────────────────────────────────────────────────────
else

JOB_NAME  = $(BASE_NAME)_train
WALLTIME  = 24:00:00
CPUS      = 8
MEM       = 64G

BATCH_SIZE = 16           
LR         = 0.002
WEIGHT_DECAY = 0.05     
EPOCHS     = 300
#PATIENCE   = 30

WEIGHTS    = $(_PRETRAIN)
SUBSET_CSV = /lustre/home/u10010021/phisat2/splits/burned/burned_train_100_global.csv

endif