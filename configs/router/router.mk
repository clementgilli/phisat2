# ==========================================
# CONFIGURATION : Router
# ==========================================

MODEL       ?=

BASE_NAME   = router_$(MODEL)
QUEUE       = gpu4_std
GPUS        = 1

TASK        = classification
DATASET     = router

DATALOADER  = router
ROOT_DIR    = /lustre/home/u10010021/phisat2/data/router

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
NUM_WORKERS = 8

_BASE        = /lustre/home/u10010021/phisat2/runs

ifeq ($(MODEL), phisatnet)
	_PRETRAIN    = $(_BASE)/pretrain_reconstruction/ssl4eo/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
else ifeq ($(MODEL), random)
	_PRETRAIN    =
	override MODEL		 = phisatnet
else
	_PRETRAIN    = $(_BASE)/knowledge_distillation/ssl4eo/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt
endif

# ─── eval / submit-eval ───────────────────────────────────────────────────────
ifneq ($(filter eval submit-eval, $(MAKECMDGOALS) $(TARGET)),)

JOB_NAME  = $(BASE_NAME)_eval
WALLTIME  = 02:00:00
CPUS      = 8
MEM       = 64gb

CKPT_PATH = /lustre/home/u10010021/phisat2/runs/classification/router/$(MODEL)/full_dataset/seed_42/checkpoints/best-v1.ckpt

# ─── train / submit-train ─────────────────────────────────────────────────────
else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 12:00:00
CPUS       = 16
MEM        = 128gb

BATCH_SIZE = 128
LR         = 0.0003
EPOCHS     = 100
PATIENCE   = 20

WEIGHTS    = $(_PRETRAIN)
SUBSET_CSV =

endif