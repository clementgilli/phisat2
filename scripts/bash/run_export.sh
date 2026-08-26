#!/bin/bash

set -e

uv run python /lustre/home/u10010021/phisat2/scripts/export_weights.py \
    --teacher_ckpt "/lustre/home/u10010021/phisat2/runs/knowledge_distillation/ssl4eo/terramind_v1_large/full_dataset/seed_42/checkpoints/best.ckpt" \
    --student_ckpt "/lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/terramind_v1_large/full_dataset/seed_42/checkpoints/best-v1.ckpt" \
    --downstream_ckpts \
        lulc="/lustre/home/u10010021/phisat2/runs/segmentation/lulc/terramind_v1_large/full_dataset/seed_42/checkpoints/best-v1.ckpt" \
        floods="/lustre/home/u10010021/phisat2/runs/segmentation/floods/terramind_v1_large/full_dataset/seed_42/checkpoints/best-v1.ckpt" \
        burned="/lustre/home/u10010021/phisat2/runs/segmentation/burned/terramind_v1_large/full_dataset/seed_42/checkpoints/best-v1.ckpt" \
        clouds="/lustre/home/u10010021/phisat2/runs/segmentation/clouds/terramind_v1_large/full_dataset/seed_42/checkpoints/best-v1.ckpt" \
    --out_dir "/lustre/home/u10010021/phisat2/exported_weights"

echo "Exported in /lustre/home/u10010021/phisat2/exported_weights"