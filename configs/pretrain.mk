JOB_NAME = pretrain_phisatnet
QUEUE = gpu4_std
WALLTIME = 24:00:00
GPUS = 1
CPUS = 16
MEM = 256G

DEVICES = 1
PRECISION = bf16-mixed
NUM_WORKERS = 4

ROOT_DIR = /lustre/home/u10010021/phisat2/data/triplets
DATALOADER = triplets

BATCH_SIZE = 512
LR = 0.0003
EPOCHS = 300