# ==========================================
# CONFIGURATION : LULC 50-Shot
# ==========================================

BASE_NAME   = lulc_50shot
QUEUE       = gpu4_std
GPUS        = 1

TASK        = segmentation
DATASET     = lulc
MODEL       = phisatnet
DATALOADER  = downstream
ROOT_DIR    = /lustre/home/u10010021/phisat2/data/

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
NUM_WORKERS = 4

ifneq ($(filter submit-eval eval,$(MAKECMDGOALS) $(TARGET)),)

JOB_NAME   = $(BASE_NAME)_eval
WALLTIME   = 02:00:00
CPUS       = 4
MEM        = 32G
CKPT_PATH  ?= 

else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 02:00:00  
CPUS       = 8
MEM        = 64G

BATCH_SIZE = 16
LR         = 0.0001

EPOCHS     = 400

WEIGHTS    = /lustre/home/u10010021/phisat2/runs/pretrain_reconstruction/triplets/phisatnet/full_dataset/seed_42/checkpoints/best-v4.ckpt
SUBSET_CSV = /lustre/home/u10010021/phisat2/splits/lulc/lulc_train_50_global.csv

endif