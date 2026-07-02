JOB_NAME = eval-encoder
QUEUE       = gpu4_std
WALLTIME = 01:00:00
GPUS     = 1
CPUS     = 4
MEM      = 32GB

ROOT_DIR    = /lustre/home/u10010021/phisat2/data
MODEL    = phisatnet
BATCH_SIZE = 128

DATASET     = lulc