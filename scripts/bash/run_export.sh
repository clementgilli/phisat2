#!/bin/bash

set -e

uv run python /lustre/home/u10010021/phisat2/scripts/export_weights.py \
    --teacher_ckpt "/lustre/home/u10010021/phisat2/runs/knowledge_distillation/triplets/terramind_v1_large/full_dataset/seed_42/checkpoints/best.ckpt" \
    --student_ckpt "/lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/terramind_v1_large/full_dataset/seed_42/checkpoints/best.ckpt" \
    --downstream_ckpts \
        lulc="/lustre/home/u10010021/phisat2/runs/segmentation/lulc/terramind_v1_large/full_dataset/seed_42/checkpoints/best.ckpt" \
        floods="/lustre/home/u10010021/phisat2/runs/segmentation/floods/terramind_v1_large/full_dataset/seed_42/checkpoints/best.ckpt" \
        burned="/lustre/home/u10010021/phisat2/runs/segmentation/burned/terramind_v1_large/full_dataset/seed_42/checkpoints/best.ckpt" \
        clouds="/lustre/home/u10010021/phisat2/runs/segmentation/clouds/terramind_v1_large/full_dataset/seed_42/checkpoints/best.ckpt" \
        building="/lustre/home/u10010021/phisat2/runs/pixel_regression/building/terramind_v1_large/full_dataset/seed_42/checkpoints/best.ckpt" \
        roads="/lustre/home/u10010021/phisat2/runs/pixel_regression/roads/terramind_v1_large/full_dataset/seed_42/checkpoints/best.ckpt" \
        router="/lustre/home/u10010021/phisat2/runs/classification/router/terramind_v1_large/full_dataset/seed_42/checkpoints/best.ckpt" \
    --out_dir "/lustre/home/u10010021/phisat2/exported_weights"

echo "Exported in /lustre/home/u10010021/phisat2/exported_weights"