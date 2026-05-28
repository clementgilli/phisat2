#!/bin/bash

#PBS -N test-phisat
#PBS -q gpu4_std
#PBS -l walltime=24:00:00
#PBS -l select=1:ngpus=1:ncpus=64:mem=400g

# -j oe : "Join Output and Error"
#PBS -j oe
#PBS -o /lustre/home/u10010021/phisat2/logs/

export WANDB_MODE=offline

cd /lustre/home/u10010021/phisat2/ && make train SUBSET_CSV=/lustre/home/u10010021/phisat2/splits/lulc/lulc_train_50_global.csv ROOT_DIR=/lustre/home/u10010021/phisat2/data/