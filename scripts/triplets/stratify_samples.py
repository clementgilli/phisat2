#!/usr/bin/env python3
"""Stratified Train/Val/Test split of the Triplets dataset using class histograms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_CLASS_PREFIX = "class_"
TILE_ID_COL = "tile_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratified Train/Val/Test split based on histograms.")
    parser.add_argument("--hist-csv", required=True, help="Input CSV produced by the extraction script.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction for training set.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction for validation set.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic selection.")
    return parser.parse_args()


def _read_histogram_csv(hist_csv: Path) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    with hist_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        
        if TILE_ID_COL not in header:
            raise ValueError(f"Missing column '{TILE_ID_COL}' in {hist_csv}")

        class_cols = [col for col in header if col.startswith(DEFAULT_CLASS_PREFIX)]
        if not class_cols:
            raise ValueError(f"No class columns found in {hist_csv}")

        rows = list(reader)

    tile_ids = [row[TILE_ID_COL] for row in rows]
    matrix = np.zeros((len(rows), len(class_cols)), dtype=np.float64)
    total_pixels = np.zeros(len(rows), dtype=np.float64)

    for i, row in enumerate(rows):
        for j, col in enumerate(class_cols):
            value = float(row.get(col, "0") or 0)
            matrix[i, j] = value
        total_pixels[i] = float(row.get("total_pixels", 0) or 0)

    return matrix, tile_ids, class_cols, total_pixels


def _target_distribution(matrix: np.ndarray) -> np.ndarray:
    full_counts = matrix.sum(axis=0)
    total = float(full_counts.sum())
    if total <= 0:
        raise ValueError("Cannot compute global target distribution: no class pixels found.")
    return full_counts / total


def _select_by_residual_matching(
    matrix: np.ndarray,
    total_pixels: np.ndarray,
    target_freq: np.ndarray,
    target_size: int,
    seed: int,
) -> np.ndarray:
    num_samples = matrix.shape[0]
    if target_size >= num_samples:
        return np.arange(num_samples)
    if num_samples == 0 or target_size <= 0:
        return np.empty(0, dtype=np.int64)

    mean_pixels = float(total_pixels.mean()) if float(total_pixels.mean()) > 0 else 1.0
    expected_total_pixels = target_size * mean_pixels
    target_counts = target_freq * expected_total_pixels

    class_orders = [np.argsort(-matrix[:, cls], kind="mergesort") for cls in range(matrix.shape[1])]
    class_ptr = np.zeros(matrix.shape[1], dtype=np.int64)

    selected_indices: list[int] = []
    selected_mask = np.zeros(num_samples, dtype=bool)
    selected_counts = np.zeros(matrix.shape[1], dtype=np.float64)

    rng = np.random.default_rng(seed)

    for _ in range(target_size):
        if len(selected_indices) >= target_size:
            break

        deficits = target_counts - selected_counts
        active = target_counts > 0
        positive = np.flatnonzero((deficits > 0) & active)
        idx = None

        if positive.size > 0:
            rel_deficits = np.zeros_like(deficits)
            rel_deficits[positive] = deficits[positive] / np.maximum(target_counts[positive], 1.0)
            
            class_order = np.argsort(-rel_deficits)
            for class_idx in class_order:
                if rel_deficits[class_idx] <= 0:
                    continue
                order = class_orders[class_idx]
                ptr = int(class_ptr[class_idx])
                while ptr < len(order) and selected_mask[order[ptr]]:
                    ptr += 1
                if ptr >= len(order):
                    class_ptr[class_idx] = ptr
                    continue
                candidate = int(order[ptr])
                class_ptr[class_idx] = ptr + 1
                idx = candidate
                break

        if idx is None:
            active_samples = np.flatnonzero(~selected_mask)
            if active_samples.size == 0:
                break
            deficits_positive = np.maximum(deficits, 0.0)
            if np.all(deficits_positive == 0):
                max_pixels = np.max(total_pixels[active_samples])
                tie_mask = np.flatnonzero(total_pixels[active_samples] == max_pixels)
                winner = int(active_samples[tie_mask[rng.integers(len(tie_mask))]])
                idx = int(winner)
            else:
                scores = matrix[active_samples] @ deficits_positive
                scores = scores + 1e-12 * rng.random(len(scores))
                idx = int(active_samples[int(np.argmax(scores))])

        selected_mask[idx] = True
        selected_indices.append(idx)
        selected_counts += matrix[idx]

    return np.array(selected_indices, dtype=np.int64)


def write_subset_csv(output_path: Path, tile_ids: list[str], indices: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([TILE_ID_COL])
        for idx in indices:
            writer.writerow([tile_ids[idx]])


def _compute_fractions(matrix: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    if len(indices) == 0:
        return {}
    counts = matrix[indices].sum(axis=0)
    total = float(counts.sum())
    if total == 0:
        return {str(i): 0.0 for i in range(len(counts))}
    return {str(i): float(counts[i] / total) for i in range(len(counts))}


def main() -> None:
    args = parse_args()
    hist_csv = Path(args.hist_csv).expanduser().resolve()
    
    matrix, tile_ids, class_cols, total_pixels = _read_histogram_csv(hist_csv)
    total_samples = matrix.shape[0]

    if args.train_ratio + args.val_ratio > 1.0:
        print("[ERROR] Train and val ratios must sum to a value <= 1.0")
        return

    train_size = int(total_samples * args.train_ratio)
    val_size = int(total_samples * args.val_ratio)

    print(f"[INFO] Stratifying {total_samples} samples...")
    target_freq = _target_distribution(matrix)

    # 1. Stratify Train
    train_idx = _select_by_residual_matching(matrix, total_pixels, target_freq, train_size, args.seed)

    # 2. Stratify Val from the remaining
    all_idx = np.arange(total_samples)
    remaining_idx = np.setdiff1d(all_idx, train_idx)
    
    matrix_rem = matrix[remaining_idx]
    total_pixels_rem = total_pixels[remaining_idx]
    
    val_sub_idx = _select_by_residual_matching(matrix_rem, total_pixels_rem, target_freq, val_size, args.seed + 1)
    val_idx = remaining_idx[val_sub_idx]

    # 3. Test is whatever is left
    test_idx = np.setdiff1d(remaining_idx, val_idx)

    # Auto-generate output filenames based on input
    train_csv = hist_csv.with_name(f"{hist_csv.stem}_train.csv")
    val_csv = hist_csv.with_name(f"{hist_csv.stem}_val.csv")
    test_csv = hist_csv.with_name(f"{hist_csv.stem}_test.csv")
    summary_json = hist_csv.with_name(f"{hist_csv.stem}_split_summary.json")

    write_subset_csv(train_csv, tile_ids, train_idx)
    write_subset_csv(val_csv, tile_ids, val_idx)
    write_subset_csv(test_csv, tile_ids, test_idx)

    # Compile Summary
    target_fractions = {col: float(target_freq[i]) for i, col in enumerate(class_cols)}
    train_fractions = _compute_fractions(matrix, train_idx)
    val_fractions = _compute_fractions(matrix, val_idx)
    test_fractions = _compute_fractions(matrix, test_idx)

    summary = {
        "input_csv": str(hist_csv),
        "seed": args.seed,
        "total_samples": total_samples,
        "target_distribution": target_fractions,
        "splits": {
            "train": {
                "count": len(train_idx),
                "file": train_csv.name,
                "distribution": {class_cols[int(k)]: v for k, v in train_fractions.items()}
            },
            "val": {
                "count": len(val_idx),
                "file": val_csv.name,
                "distribution": {class_cols[int(k)]: v for k, v in val_fractions.items()}
            },
            "test": {
                "count": len(test_idx),
                "file": test_csv.name,
                "distribution": {class_cols[int(k)]: v for k, v in test_fractions.items()}
            }
        }
    }

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"[INFO] Successfully stratified dataset.")
    print(f"[INFO] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"[INFO] Summary saved to: {summary_json}")


if __name__ == "__main__":
    
    main()