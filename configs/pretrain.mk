# ==========================================
# CONFIGURATION : Pretrain SSL (Triplets)
# ==========================================

BASE_NAME  = pretrain_phisatnet
QUEUE      = gpu4_std
GPUS       = 1

TASK       = pretrain_reconstruction
DATASET    = triplets
MODEL      = phisatnet
DATALOADER = triplets
ROOT_DIR   = /lustre/home/u10010021/phisat2/data/triplets

SEEDS      = 42
DEVICES    = 1
PRECISION  = bf16-mixed
BATCH_SIZE = 512
NUM_WORKERS = 4

ifneq ($(filter submit-eval eval,$(MAKECMDGOALS) $(TARGET)),)

JOB_NAME   = $(BASE_NAME)_eval
WALLTIME   = 03:00:00
CPUS       = 8
MEM        = 64G
CKPT_PATH  ?= 

else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 24:00:00
CPUS       = 16
MEM        = 256G

EPOCHS     = 300
LR         = 0.0003

endif