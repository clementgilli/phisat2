# ==========================================
# CONFIGURATION : Domain Adaptation & Gap Eval
# ==========================================

BASE_NAME   = da_phisatnet
QUEUE       = gpu4_std
GPUS        = 1

MODEL       = phisatnet
ROOT_DIR    = /lustre/home/u10010021/phisat2/data/triplets

SEEDS       = 42 7 6
DEVICES     = 1
PRECISION   = bf16-mixed
BATCH_SIZE  = 128
NUM_WORKERS = 14

# ── Paths ────────────────────────────
_BASE        = /lustre/home/u10010021/phisat2/runs
_PRETRAIN    = $(_BASE)/pretrain_reconstruction/triplets/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt
_LULC        = $(_BASE)/segmentation/lulc/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
_FLOODS      = $(_BASE)/segmentation/floods/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt
_CLOUDS      = $(_BASE)/segmentation/clouds/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
_BURNED      = $(_BASE)/segmentation/burned/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt
_ROADS       = $(_BASE)/pixel_regression/roads/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt
_BLDG        = $(_BASE)/pixel_regression/building/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt
_ROUTER      = $(_BASE)/classification/router/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

# ─── eval-domain-gap / submit-eval-domain-gap ────────────────────────────────
ifneq ($(filter eval-domain-gap submit-eval-domain-gap, $(MAKECMDGOALS) $(TARGET)),)

JOB_NAME  = $(BASE_NAME)_eval
WALLTIME  = 03:00:00
CPUS      = 8
MEM       = 64G

# Teacher = checkpoint SSL (always the same, frozen)
TEACHER_CKPT = $(_PRETRAIN)

# Student = checkpoint after DA if CKPT_PATH,
# else build_model on TEACHER_CKPT (eval baseline pre-DA).
STUDENT_CKPT = $(CKPT_PATH)

DECODERS = lulc=$(_LULC) floods=$(_FLOODS) clouds=$(_CLOUDS) burned=$(_BURNED) roads=$(_ROADS) building=$(_BLDG) router=$(_ROUTER)

# ─── domain-adaptation / submit-domain-adaptation ────────────────────────────
else

JOB_NAME  = $(BASE_NAME)_train
WALLTIME  = 24:00:00
CPUS      = 16
MEM       = 256G

TASK    = domain_adaptation
EPOCHS  = 300
PATIENCE = 50
LR      = 0.001

# Initialisation weights (teacher and student)
WEIGHTS = $(_PRETRAIN)

endif