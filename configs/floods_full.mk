# ==========================================
# CONFIGURATION : Floods Full Dataset
# ==========================================

BASE_NAME   = floods_full
QUEUE       = gpu4_std
GPUS        = 1

TASK        = segmentation
DATASET     = floods
MODEL       = phisatnet
DATALOADER  = downstream
ROOT_DIR    = /lustre/home/u10010021/phisat2/data/

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
NUM_WORKERS = 8

ifneq ($(filter submit-eval eval,$(MAKECMDGOALS) $(TARGET)),)

JOB_NAME   = $(BASE_NAME)_eval
WALLTIME   = 02:00:00
CPUS       = 8
MEM        = 64G
CKPT_PATH  ?= 

else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 12:00:00  
CPUS       = 16
MEM        = 128G

BATCH_SIZE = 128

LR         = 0.0003
EPOCHS     = 100

WEIGHTS    = /lustre/home/u10010021/phisat2/runs/pretrain_reconstruction/triplets/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt
SUBSET_CSV = 

endif