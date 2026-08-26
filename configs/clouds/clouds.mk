# ==========================================
# CONFIGURATION : Clouds — N-Shot Experiments
# ==========================================

MODEL       ?=
NSHOT       = full

BASE_NAME   = clouds_$(MODEL)_$(NSHOT)
QUEUE       = gpu4_std
GPUS        = 1

TASK        = segmentation
DATASET     = clouds
DATALOADER  = downstream_s2
ROOT_DIR    = /lustre/home/u10010021/phisat2/data

BASE_CHANNELS = 16

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

ifeq ($(NSHOT), full)
	CKPT_PATH = $(_BASE)/segmentation/clouds/$(MODEL)/full_dataset/seed_42/checkpoints/best-v1.ckpt
else
	CKPT_PATH = $(_BASE)/segmentation/clouds/$(MODEL)/$(DATASET)_split_$(NSHOT)/seed_42/checkpoints/best.ckpt
endif

# ─── train / submit-train ─────────────────────────────────────────────────────
else

JOB_NAME   = $(BASE_NAME)_train
WALLTIME   = 12:00:00
CPUS       = 16
MEM        = 128gb
WEIGHTS    = $(_PRETRAIN)

ifeq ($(NSHOT), 100)
    EPOCHS     = 600
	PATIENCE   = 60
	BATCH_SIZE   = 16
    LR           = 0.00005
    WEIGHT_DECAY = 0.005
	
    SUBSET_CSV   = $(ROOT_DIR)/$(DATASET)/$(DATASET)_split_100.csv

else ifeq ($(NSHOT), 1000)
	EPOCHS     = 400
	PATIENCE   = 40
    BATCH_SIZE   = 64
    LR           = 0.00015
    WEIGHT_DECAY = 0.001
    SUBSET_CSV   = $(ROOT_DIR)/$(DATASET)/$(DATASET)_split_1000.csv

else ifeq ($(NSHOT), 10000)
	EPOCHS     = 300
	PATIENCE   = 20
    BATCH_SIZE   = 128
    LR           = 0.0003
    WEIGHT_DECAY = 0.0001
    SUBSET_CSV   = $(ROOT_DIR)/$(DATASET)/$(DATASET)_split_10000.csv

else
   	EPOCHS     = 150
	PATIENCE   = 20
	BATCH_SIZE = 128
	LR         = 0.0003
	WEIGHT_DECAY = 0.0001
    SUBSET_CSV   = 
	
endif

endif