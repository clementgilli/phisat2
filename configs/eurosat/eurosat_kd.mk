# ==========================================
# CONFIGURATION : EuroSAT
# ==========================================

MODEL       ?=

BASE_NAME   = eurosat_$(MODEL)
QUEUE       = gpu4_std
GPUS        = 1

TASK        = classification
DATASET     = eurosat

DATALOADER  = eurosat
ROOT_DIR    = /lustre/home/u10010021/phisat2/data

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
NUM_WORKERS = 8

_PRETRAIN = /lustre/home/u10010021/phisat2/runs/knowledge_distillation/triplets/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt

# ─── eval / submit-eval ───────────────────────────────────────────────────────
ifneq ($(filter eval submit-eval, $(MAKECMDGOALS) $(TARGET)),)

JOB_NAME  = $(BASE_NAME)_eval
WALLTIME  = 02:00:00
CPUS      = 8
MEM       = 64gb

CKPT_PATH = /lustre/home/u10010021/phisat2/runs/classification/eurosat/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt

# ─── train / submit-train ─────────────────────────────────────────────────────
else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 12:00:00
CPUS       = 16
MEM        = 128gb

BATCH_SIZE = 128
LR         = 0.0003
EPOCHS     = 100

WEIGHTS    = $(_PRETRAIN)
SUBSET_CSV =

endif