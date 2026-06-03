# ==========================================
# CONFIGURATION : LULC 50-Shot
# ==========================================

# --- Shared parameters (Train & Eval) ---
BASE_NAME   = lulc_50_shot
QUEUE      = gpu4_std
GPUS       = 1
TASK       = segmentation
DATASET    = lulc
MODEL      = terramind_v1_tiny
BATCH_SIZE = 32
SEEDS      = 42
ROOT_DIR   = /lustre/home/u10010021/phisat2/data/
DATALOADER = zarr_downstream

# ==========================================
ifneq ($(filter submit-eval eval,$(MAKECMDGOALS) $(TARGET)),)

# Eval
JOB_NAME  = $(BASE_NAME)_eval
WALLTIME  = 03:00:00
CPUS      = 8
MEM       = 50g
CKPT_PATH ?= 

else

# Train
JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 24:00:00
CPUS       = 4
MEM        = 100g

EPOCHS     = 100
LR         = 0.05
SUBSET_CSV = /lustre/home/u10010021/phisat2/splits/lulc/lulc_train_50_global.csv

endif