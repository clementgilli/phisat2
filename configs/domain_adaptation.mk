# ==========================================
# CONFIGURATION : Domain Adaptation & Gap Eval
# ==========================================

BASE_NAME  = da_phisatnet
QUEUE      = gpu4_std
GPUS       = 1

TASK       = domain_adaptation
DATASET    = triplets
MODEL      = phisatnet
DATALOADER = triplets
ROOT_DIR   = /lustre/home/u10010021/phisat2/data/triplets

SEEDS      = 42
DEVICES    = 1
PRECISION  = bf16-mixed
BATCH_SIZE = 16
NUM_WORKERS = 4

ifneq ($(filter submit-eval eval,$(MAKECMDGOALS) $(TARGET)),)

JOB_NAME   = $(BASE_NAME)_eval
WALLTIME   = 03:00:00
CPUS       = 8
MEM        = 64G

TASK       = eval_domain_gap

CKPT_PATH  ?= 

TEACHER_CKPT = /lustre/home/u10010021/phisat2/runs/pretrain_reconstruction/triplets/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

LULC_CKPT    = /lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
FLOODS_CKPT  = /lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
BURNED_CKPT  = /lustre/home/u10010021/phisat2/runs/segmentation/burned/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
ROADS_CKPT   = /lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
BLDG_CKPT    = /lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt

DECODERS_LIST = lulc=$(LULC_CKPT) floods=$(FLOODS_CKPT) burned=$(BURNED_CKPT) roads=$(ROADS_CKPT) building=$(BLDG_CKPT)

EXTRA_ARGS := --teacher_ckpt $(TEACHER_CKPT)

ifneq ($(strip $(CKPT_PATH)),)
    EXTRA_ARGS += --student_ckpt $(CKPT_PATH)
endif

EXTRA_ARGS += --decoders $(DECODERS_LIST)

else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 24:00:00
CPUS       = 16
MEM        = 256G

EPOCHS     = 300
LR         = 0.0003

WEIGHTS    = /lustre/home/u10010021/phisat2/runs/pretrain_reconstruction/triplets/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

endif