# ==========================================
# CONFIGURATION : Knowledge Distillation
# ==========================================

MODEL      ?=

BASE_NAME   = kd_${MODEL}
QUEUE       = gpu4_std
GPUS        = 1

#ROOT_DIR    = /lustre/home/u10010021/phisat2/data/triplets
ROOT_DIR    = /lustre/home/u10010021/phisat2/data/ssl4eo
DATASET    = ssl4eo
DATALOADER   = ssl4eo

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
BATCH_SIZE  = 128
NUM_WORKERS = 14

JOB_NAME  = $(BASE_NAME)_${MODEL}
WALLTIME  = 24:00:00
CPUS      = 16
MEM       = 256gb

TASK    = knowledge_distillation
EPOCHS  = 300
#PATIENCE = 50
LR      = 0.001
WEIGHT_DECAY = 0.0001

BASE_CHANNELS = 16
