# ==========================================
# CONFIGURATION : Clouds — Full Dataset
# ==========================================

MODEL       ?=

BASE_NAME   = clouds_$(MODEL)_full
QUEUE       = gpu4_std
GPUS        = 1

TASK        = segmentation
DATASET     = clouds
DATALOADER  = downstream
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
MEM       = 64G

CKPT_PATH = /lustre/home/u10010021/phisat2/runs/segmentation/clouds/$(MODEL)/clouds_500shot/seed_42/checkpoints/best.ckpt

# ─── train / submit-train ─────────────────────────────────────────────────────
else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 12:00:00
CPUS       = 16
MEM        = 128G

BATCH_SIZE = 32
LR         = 0.005
WEIGHT_DECAY = 0.05
EPOCHS     = 200

WEIGHTS    = $(_PRETRAIN)
SUBSET_CSV = /lustre/home/u10010021/phisat2/splits/clouds/clouds_500shot.csv

endif