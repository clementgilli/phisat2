# ==========================================
# CONFIGURATION : Domain Adaptation & Gap Eval
# ==========================================

MODEL       = terramind_v1_large

BASE_NAME   = da_$(MODEL)
QUEUE       = gpu4_std
GPUS        = 1

ROOT_DIR    = /lustre/home/u10010021/phisat2/data/triplets

SEEDS       = 42
DEVICES     = 1
PRECISION   = bf16-mixed
BATCH_SIZE  = 128
NUM_WORKERS = 14

# ── Paths ────────────────────────────
_BASE        = /lustre/home/u10010021/phisat2/runs

ifeq ($(MODEL), phisatnet)
	_PRETRAIN    = $(_BASE)/pretrain_reconstruction/ssl4eo/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt
else
	_PRETRAIN    = $(_BASE)/knowledge_distillation/ssl4eo/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt
endif

#_LULC        = $(_BASE)/segmentation/lulc/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt
#_FLOODS      = $(_BASE)/segmentation/floods/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt
#_CLOUDS      = $(_BASE)/segmentation/clouds/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt
#_BURNED      = $(_BASE)/segmentation/burned/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt
#_ROADS       = $(_BASE)/pixel_regression/roads/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt
#_BLDG        = $(_BASE)/pixel_regression/building/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt
#_ROUTER      = $(_BASE)/classification/router/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt

#_METHANE	 = $(_BASE)/segmentation/methane/$(MODEL)/full_dataset/seed_42/checkpoints/best-v1.ckpt
_MARINE		 = $(_BASE)/segmentation/marine/$(MODEL)/full_dataset/seed_42/checkpoints/best-v6.ckpt
_LULC		 = $(_BASE)/segmentation/lulc/$(MODEL)/full_dataset/seed_42/checkpoints/best-v6.ckpt
_CLOUDS		 = $(_BASE)/segmentation/clouds/$(MODEL)/full_dataset/seed_42/checkpoints/best-v4.ckpt
_FLOODS		 = $(_BASE)/segmentation/floods/$(MODEL)/full_dataset/seed_42/checkpoints/best-v3.ckpt

# ─── eval-domain-gap / submit-eval-domain-gap ────────────────────────────────
ifneq ($(filter eval-domain-gap submit-eval-domain-gap, $(MAKECMDGOALS) $(TARGET)),)

JOB_NAME  = $(BASE_NAME)_eval
WALLTIME  = 03:00:00
CPUS      = 8
MEM       = 64gb

# Teacher = checkpoint SSL (always the same, frozen)
TEACHER_CKPT = $(_PRETRAIN)

# Student = checkpoint after DA if CKPT_PATH,
# else build_model on TEACHER_CKPT (eval baseline pre-DA).
STUDENT_CKPT = $(CKPT_PATH)

DECODERS = lulc=$(_LULC) floods=$(_FLOODS) clouds=$(_CLOUDS) marine=$(_MARINE) #burned=$(_BURNED) roads=$(_ROADS) building=$(_BLDG) router=$(_ROUTER) 

# ─── domain-adaptation / submit-domain-adaptation ────────────────────────────
else

JOB_NAME  = $(BASE_NAME)_train
WALLTIME  = 24:00:00
CPUS      = 16
MEM       = 256gb

TASK    = domain_adaptation
EPOCHS  = 300
PATIENCE = 50
LR      = 0.001
WEIGHT_DECAY = 0.0001

# Initialisation weights (teacher and student)
WEIGHTS = $(_PRETRAIN)

endif