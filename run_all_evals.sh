#!/bin/bash

set -e

source .venv/bin/activate

make submit-eval EXPERIMENT=configs/lulc_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt

make submit-eval EXPERIMENT=configs/floods_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt

make submit-eval EXPERIMENT=configs/burned_area_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/burned/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt

make submit-eval EXPERIMENT=configs/roads_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt

make submit-eval EXPERIMENT=configs/building_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt

#echo "-> Fire..."
#make submit-eval EXPERIMENT=configs/fire_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/classification/fire/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt

make submit-eval EXPERIMENT=configs/domain_adaptation.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/phisatnet/full_dataset/seed_42/checkpoints/best-v5.ckpt

wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/full_dataset/seed_42/wandb/offline-run-20260610_223032-6cd2xsuk
wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/full_dataset/seed_42/wandb/offline-run-20260610_223052-lpawbtmg
wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/burned/phisatnet/full_dataset/seed_42/wandb/offline-run-20260610_223035-wfymgb5t
wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/full_dataset/seed_42/wandb/offline-run-20260610_223042-y9y6i3d8
wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/full_dataset/seed_42/wandb/offline-run-20260610_223039-vnniu4y3
wandb sync /lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/phisatnet/full_dataset/seed_42/wandb/offline-run-20260610_223353-ekdpl6om


echo "Done."