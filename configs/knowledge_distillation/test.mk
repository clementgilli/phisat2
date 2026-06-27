# ==========================================
# CONFIGURATION : Knowledge Distillation
# ==========================================

BASE_NAME   = kd_test
QUEUE       = gpu4_std
GPUS        = 1

MODEL       = seco_resnet50_sentinel2_rgb_seco
ROOT_DIR    = /lustre/home/u10010021/phisat2/data/triplets

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
BATCH_SIZE  = 128
NUM_WORKERS = 14

JOB_NAME  = $(BASE_NAME)_train
WALLTIME  = 24:00:00
CPUS      = 16
MEM       = 256G

TASK    = knowledge_distillation
EPOCHS  = 2
PATIENCE = 50
LR      = 0.001