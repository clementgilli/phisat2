#!/usr/bin/env python3
"""Select segmentation sample subsets from precomputed per-sample class histograms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_CLASS_PREFIX = "class_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratify segmentation samples from per-sample class histograms.")
    parser.add_argument("--hist-csv", required=True, help="Input CSV produced by build_sample_histograms.py.")
    parser.add_argument(
        "--train-size",
        required=True,
        help="Target train subset size (integer) or full.",
    )
    parser.add_argument(
        "--strategy",
        default="global",
        choices=("global", "balanced"),
        help="global keeps full-train class distribution; balanced targets uniform class frequencies.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic tie-breaks.")
    parser.add_argument(
        "--sample-id-col",
        default="sample_id",
        help="Column name containing sample identifiers.",
    )
    parser.add_argument("--class-prefix", default=DEFAULT_CLASS_PREFIX, help="Prefix for class columns in CSV.")
    parser.add_argument(
        "--selected-csv",
        default=None,
        help="Output CSV with selected sample ids (default auto-generated from input file).",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Output JSON summary (default auto-generated from input file).",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional per-class summary CSV path.",
    )
    parser.add_argument(
        "--full-columns",
        default=None,
        help="Optional comma-separated list of target class columns; defaults to all class_* columns in input.",
    )
    return parser.parse_args()


def _read_histogram_csv(
    hist_csv: Path,
    sample_id_col: str,
    class_prefix: str,
    full_columns: str | None = None,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    with hist_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if sample_id_col not in header:
            raise ValueError(f"Missing sample-id column '{sample_id_col}' in {hist_csv}")

        if full_columns:
            class_cols = [col.strip() for col in full_columns.split(",") if col.strip()]
        else:
            class_cols = [col for col in header if col.startswith(class_prefix)]
        if not class_cols:
            raise ValueError(f"No class columns found in {hist_csv}")

        rows = list(reader)

    sample_ids = [row[sample_id_col] for row in rows]
    matrix = np.zeros((len(rows), len(class_cols)), dtype=np.float64)
    total_pixels = np.zeros(len(rows), dtype=np.float64)

    for i, row in enumerate(rows):
        for j, col in enumerate(class_cols):
            value = float(row.get(col, "0") or 0)
            matrix[i, j] = value
        total_pixels[i] = float(row.get("total_pixels", 0) or 0)

    return matrix, sample_ids, class_cols, total_pixels


def _parse_train_size(requested: str, total_samples: int) -> int:
    if requested == "full":
        return total_samples
    try:
        value = int(requested)
    except ValueError as exc:
        raise ValueError(f"--train-size must be 'full' or an integer, got: {requested}") from exc
    if value <= 0:
        raise ValueError("--train-size must be positive")
    return min(value, total_samples)


def _target_distribution(matrix: np.ndarray, strategy: str) -> np.ndarray:
    full_counts = matrix.sum(axis=0)
    if strategy == "global":
        total = float(full_counts.sum())
        if total <= 0:
            raise ValueError("Cannot compute global target distribution: no class pixels found.")
        return full_counts / total

    active = full_counts > 0
    if not np.any(active):
        raise ValueError("Cannot compute balanced target distribution: no class pixels found.")
    num_active = int(active.sum())
    dist = np.zeros_like(full_counts, dtype=np.float64)
    dist[active] = 1.0 / num_active
    return dist


def _compute_class_counts(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    return matrix[indices].sum(axis=0)


def _select_by_residual_matching(
    matrix: np.ndarray,
    total_pixels: np.ndarray,
    target_freq: np.ndarray,
    train_size: int,
    seed: int,
) -> np.ndarray:
    num_samples = matrix.shape[0]
    if train_size >= num_samples:
        return np.arange(num_samples)
    if num_samples == 0 or train_size <= 0:
        return np.empty(0, dtype=np.int64)

    mean_pixels = float(total_pixels.mean()) if float(total_pixels.mean()) > 0 else 1.0
    expected_total_pixels = train_size * mean_pixels
    target_counts = target_freq * expected_total_pixels

    # Per-class ranking (descending count) for each class.
    class_orders = [np.argsort(-matrix[:, cls], kind="mergesort") for cls in range(matrix.shape[1])]
    class_ptr = np.zeros(matrix.shape[1], dtype=np.int64)

    selected_indices: list[int] = []
    selected_mask = np.zeros(num_samples, dtype=bool)
    selected_counts = np.zeros(matrix.shape[1], dtype=np.float64)

    rng = np.random.default_rng(seed)

    for _ in range(train_size):
        if len(selected_indices) >= train_size:
            break

        deficits = target_counts - selected_counts
        active = target_counts > 0
        positive = np.flatnonzero((deficits > 0) & active)
        idx = None

        if positive.size > 0:
            rel_deficits = np.zeros_like(deficits)
            rel_deficits[positive] = deficits[positive] / np.maximum(target_counts[positive], 1.0)
            # choose class with largest relative deficit and take next not-yet-selected sample
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
            # Fallback: pick the not-selected sample that best matches the current residual.
            active_samples = np.flatnonzero(~selected_mask)
            if active_samples.size == 0:
                break
            deficits_positive = np.maximum(deficits, 0.0)
            if np.all(deficits_positive == 0):
                # Deterministic random choice among equal highest-pixel ties.
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


def _build_summary(
    class_cols: list[str],
    full_counts: np.ndarray,
    subset_counts: np.ndarray,
    subset_size: int,
    target_dist: np.ndarray,
) -> dict[str, object]:
    full_total = float(full_counts.sum())
    subset_total = float(subset_counts.sum())
    full_freq = full_counts / full_total if full_total > 0 else np.zeros_like(full_counts)
    subset_freq = subset_counts / subset_total if subset_total > 0 else np.zeros_like(subset_counts)
    l1_distance = float(np.abs(subset_freq - target_dist).sum())

    return {
        "full_counts": {col: int(full_counts[i]) for i, col in enumerate(class_cols)},
        "full_fractions": {col: float(full_freq[i]) for i, col in enumerate(class_cols)},
        "subset_counts": {col: int(subset_counts[i]) for i, col in enumerate(class_cols)},
        "subset_fractions": {col: float(subset_freq[i]) for i, col in enumerate(class_cols)},
        "target_fractions": {col: float(target_dist[i]) for i, col in enumerate(class_cols)},
        "subset_size": int(subset_size),
        "subset_total_pixels": float(subset_total),
        "full_total_pixels": float(full_total),
        "l1_distance": l1_distance,
    }


def write_selected_csv(
    output_path: Path,
    sample_ids: list[str],
    selected: np.ndarray,
    extra_columns: list[str],
    matrix: np.ndarray,
    total_pixels: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "sample_id", "total_pixels"] + extra_columns
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, idx in enumerate(selected.tolist(), start=1):
            row = {
                "rank": rank,
                "sample_id": sample_ids[int(idx)],
                "total_pixels": int(total_pixels[int(idx)]),
            }
            for j, col in enumerate(extra_columns):
                row[col] = int(matrix[int(idx), j])
            writer.writerow(row)


def write_summary_json(
    output_path: Path,
    hist_csv: Path,
    strategy: str,
    requested: str,
    selected: np.ndarray,
    summary: dict[str, object],
    seed: int,
) -> None:
    payload = {
        "hist_csv": str(hist_csv),
        "strategy": strategy,
        "requested_size": requested,
        "seed": seed,
        "selected_size": int(selected.size),
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_summary_csv(output_path: Path, summary: dict[str, object], class_cols: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for col in class_cols:
        rows.append(
            {
                "class": col,
                "full_count": summary["full_counts"][col],
                "subset_count": summary["subset_counts"][col],
                "full_fraction": summary["full_fractions"][col],
                "subset_fraction": summary["subset_fractions"][col],
                "target_fraction": summary["target_fractions"][col],
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class",
                "full_count",
                "subset_count",
                "full_fraction",
                "subset_fraction",
                "target_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    hist_csv = Path(args.hist_csv).expanduser().resolve()
    matrix, sample_ids, class_cols, total_pixels = _read_histogram_csv(
        hist_csv,
        sample_id_col=args.sample_id_col,
        class_prefix=args.class_prefix,
        full_columns=args.full_columns,
    )

    train_size = _parse_train_size(args.train_size, matrix.shape[0])
    target_freq = _target_distribution(matrix, args.strategy)
    selected = _select_by_residual_matching(matrix, total_pixels, target_freq, train_size, args.seed)

    full_counts = matrix.sum(axis=0)
    selected_counts = _compute_class_counts(matrix, selected)
    summary = _build_summary(class_cols, full_counts, selected_counts, selected.size, target_freq)

    selected_csv = Path(args.selected_csv) if args.selected_csv else hist_csv.with_name(f"{hist_csv.stem}.{args.strategy}.{len(selected)}.selected.csv")
    summary_json = Path(args.summary_json) if args.summary_json else hist_csv.with_name(f"{hist_csv.stem}.{args.strategy}.{len(selected)}.summary.json")
    write_selected_csv(selected_csv, sample_ids, selected, class_cols, matrix, total_pixels)
    write_summary_json(summary_json, hist_csv, args.strategy, args.train_size, selected, summary, args.seed)

    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), summary, class_cols)

    print(f"[OK] Selected {len(selected)} samples with '{args.strategy}' strategy -> {selected_csv}")
    print(f"[OK] Summary saved to {summary_json}")


if __name__ == "__main__":
    main()
