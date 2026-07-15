JOB_NAME = eval-encoder
QUEUE       = gpu4_std
WALLTIME = 01:00:00
GPUS     = 1
CPUS     = 4
MEM      = 32GB

ROOT_DIR    = /lustre/home/u10010021/phisat2/data
WEIGHTS  = /lustre/home/u10010021/phisat2/runs/knowledge_distillation/triplets/$(MODEL)/full_dataset/seed_42/checkpoints/best.ckpt
BATCH_SIZE = 128

MODEL    ?=

BASE_CHANNELS = 16
DATASET = router