#!/usr/bin/env python3
"""Create stratified N-shot subsets from an existing Train split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_CLASS_PREFIX = "class_"
TILE_ID_COL = "tile_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stratified N-shot subsets from a train split.")
    parser.add_argument("--hist-csv", required=True, help="Input CSV with class histograms.")
    parser.add_argument("--train-csv", required=True, help="Input CSV containing the full train split.")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10, 50], 
                        help="List of subset sizes to generate (e.g., 1 5 10).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic selection.")
    return parser.parse_args()


def _get_train_tiles(train_csv: Path) -> set[str]:
    with train_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if TILE_ID_COL not in reader.fieldnames:
            raise ValueError(f"Missing column '{TILE_ID_COL}' in {train_csv}")
        return {row[TILE_ID_COL] for row in reader}


def _read_filtered_histogram_csv(hist_csv: Path, valid_tiles: set[str]) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    with hist_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        
        if TILE_ID_COL not in header:
            raise ValueError(f"Missing column '{TILE_ID_COL}' in {hist_csv}")

        class_cols = [col for col in header if col.startswith(DEFAULT_CLASS_PREFIX)]
        if not class_cols:
            raise ValueError(f"No class columns found in {hist_csv}")

        rows = [row for row in reader if row[TILE_ID_COL] in valid_tiles]

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


def _select_best_single_image(matrix: np.ndarray, target_freq: np.ndarray) -> np.ndarray:
    
    
    row_sums = matrix.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    img_freqs = matrix / safe_sums

    distances = np.sum((img_freqs - target_freq) ** 2, axis=1)    
    best_idx = int(np.argmin(distances))
    return np.array([best_idx], dtype=np.int64)


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
    train_csv = Path(args.train_csv).expanduser().resolve()
    
    print(f"[INFO] Loading valid train tiles from: {train_csv.name}")
    train_tiles = _get_train_tiles(train_csv)
    
    print(f"[INFO] Filtering histograms for the {len(train_tiles)} train tiles...")
    matrix, tile_ids, class_cols, total_pixels = _read_filtered_histogram_csv(hist_csv, train_tiles)
    total_train_samples = matrix.shape[0]
    
    if total_train_samples == 0:
        print("[ERROR] No matching tiles found between the histogram CSV and the train CSV.")
        return

    target_freq = _target_distribution(matrix)
    target_fractions = {col: float(target_freq[i]) for i, col in enumerate(class_cols)}

    summary = {
        "input_train_csv": str(train_csv),
        "total_train_samples": total_train_samples,
        "target_train_distribution": target_fractions,
        "n_shots_generated": {}
    }

    for shot in args.shots:
        if shot > total_train_samples:
            print(f"[WARN] Requested {shot}-shot but only {total_train_samples} available. Skipping.")
            continue
            
        print(f"[INFO] Generating stratified {shot}-shot subset...")
        
        if shot == 1:
            subset_idx = _select_best_single_image(matrix, target_freq)
        else:
            subset_idx = _select_by_residual_matching(matrix, total_pixels, target_freq, shot, args.seed)
        
        subset_fractions = _compute_fractions(matrix, subset_idx)
        subset_filename = train_csv.with_name(f"{train_csv.stem}_{shot}shot.csv")
        
        write_subset_csv(subset_filename, tile_ids, subset_idx)
        
        summary["n_shots_generated"][f"{shot}_shot"] = {
            "count": len(subset_idx),
            "file": subset_filename.name,
            "distribution": {class_cols[int(k)]: v for k, v in subset_fractions.items()}
        }

    summary_json = train_csv.with_name(f"{train_csv.stem}_nshot_summary.json")
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"[INFO] Successfully generated subsets. Summary saved to: {summary_json.name}")


if __name__ == "__main__":
    main()