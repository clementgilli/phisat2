# ==========================================
# CONFIGURATION : LULC 50-Shot
# ==========================================

# --- 1. Cluster settings ---
JOB_NAME = lulc_50_shot
QUEUE    = gpu4_std
WALLTIME = 24:00:00
GPUS     = 1
CPUS     = 64
MEM      = 400g

# --- 2. Training parameters ---
TASK       = segmentation
DATASET    = lulc
MODEL      = phisat2_geoaware
EPOCHS     = 100
BATCH_SIZE = 32
LR         = 0.05
SEEDS      = 42

# --- 3. Paths ---
ROOT_DIR   = /lustre/home/u10010021/phisat2/data/
SUBSET_CSV = /lustre/home/u10010021/phisat2/splits/lulc/lulc_train_50_global.csv