# ==========================================
# CONFIGURATION : Phase 1 — SSL Pretraining
# ==========================================

BASE_NAME   = pretrain_phisatnet
QUEUE       = gpu4_std
GPUS        = 1

MODEL       = phisatnet
ROOT_DIR    = /lustre/home/u10010021/phisat2/data/triplets

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
BATCH_SIZE  = 512
NUM_WORKERS = 8

# ─── eval / submit-eval ───────────────────────────────────────────────────────
ifneq ($(filter eval submit-eval, $(MAKECMDGOALS) $(TARGET)),)

JOB_NAME  = $(BASE_NAME)_eval
WALLTIME  = 02:00:00
CPUS      = 8
MEM       = 64G

TASK      = pretrain_reconstruction
CKPT_PATH ?=

# ─── pretrain / submit-pretrain ───────────────────────────────────────────────
else

JOB_NAME  = $(BASE_NAME)_train
WALLTIME  = 24:00:00
CPUS      = 16
MEM       = 256G

EPOCHS    = 300
LR        = 0.0003
WEIGHT_DECAY = 0.05

endif