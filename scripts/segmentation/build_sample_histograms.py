#!/usr/bin/env python3
"""Build per-sample class histograms for segmentation datasets.

The output is a CSV with one row per sample and one ``class_*`` column per class
observed in the split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import zarr

DEFAULT_CLASS_PREFIX = "class_"


ZARR_DATASET_NAMES = {
    "burned": ("burned_area", "burned"),
    "floods": ("worldfloods", "floods"),
    "worldfloods": ("worldfloods", "floods"),
    "lc": ("phileo-bench_lc", "lc", "lulc"),
    "lulc": ("phileo-bench_lc", "lulc"),
    "marine": ("marine_area", "marine"),
}


def _resolve_base_path(root_dir: Path, dataset: str) -> Path:
    if root_dir.suffix == ".zarr":
        return root_dir
    dataset_names = ZARR_DATASET_NAMES.get(dataset, (dataset,))
    for dataset_name in dataset_names:
        candidate = root_dir / f"{dataset_name}.zarr"
        if candidate.exists():
            return candidate
    return root_dir / f"{dataset}.zarr"


def _list_patch_dirs(source_folder: Path) -> list[Path]:
    paths: list[Path] = []
    with os.scandir(source_folder) as entries:
        for entry in entries:
            try:
                is_dir = entry.is_dir()
            except OSError:
                continue
            if is_dir:
                paths.append(Path(entry.path))
    return sorted(paths)


def _open_array(array_path: Path):
    for attempt in range(3):
        try:
            return zarr.open_array(str(array_path), mode="r", zarr_format=3)
        except (FileNotFoundError, ValueError, OSError):
            if attempt == 2:
                break
            time.sleep(0.2 * (attempt + 1))
    return zarr.open_array(str(array_path), mode="r")


def _open_array_or_none(array_path: Path):
    try:
        return _open_array(array_path)
    except OSError:
        return None


def _read_mask(mask_path: Path) -> np.ndarray:
    array = _open_array_or_none(mask_path)
    if array is None:
        raise OSError(f"Could not open mask array at {mask_path}")
    data = np.asarray(array)
    if data.ndim == 3 and data.shape[0] == 1:
        data = np.squeeze(data, axis=0)
    return data


def _accumulate_histogram(values: np.ndarray, class_counts: Mapping[int, int], ignore_negative: bool = True) -> None:
    if values.size == 0:
        return
    flat = np.ravel(values)
    if np.issubdtype(flat.dtype, np.floating):
        flat = flat[~np.isnan(flat)]
    if np.issubdtype(flat.dtype, np.floating):
        flat = np.rint(flat).astype(np.int64, copy=False)
    else:
        flat = flat.astype(np.int64, copy=False)
    if ignore_negative:
        flat = flat[flat >= 0]
    if flat.size == 0:
        return
    unique, counts = np.unique(flat, return_counts=True)
    for cls, count in zip(unique, counts):
        class_counts[int(cls)] += int(count)


def _coerce_int(value: object) -> int:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return 0
    return int(str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-sample class histograms for a segmentation zarr dataset.")
    parser.add_argument("--dataset", required=True, help="Segmentation dataset name or explicit .zarr path.")
    parser.add_argument("--root-dir", default=".", help="Root folder containing dataset zarr stores.")
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "val", "test"),
        help="Which split to read. For train/val this resolves to trainval.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for the number of samples to process.",
    )
    parser.add_argument("--label-key", default="label", help="Label array name inside each patch group.")
    parser.add_argument("--class-prefix", default=DEFAULT_CLASS_PREFIX, help="Prefix for class columns in output CSV.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output path for per-sample histogram CSV. Default uses dataset and split naming.",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON sidecar including class columns and totals.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Log progress every N samples while reading histograms.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for histogram extraction.",
    )
    return parser.parse_args()


def _read_sample_histogram(sample_path: str, label_key: str) -> tuple[str, int, dict[int, int], bool]:
    sample_name = Path(sample_path).name
    try:
        label = _read_mask(Path(sample_path) / label_key)
    except OSError:
        return sample_name, 0, {}, False
    class_counts: dict[int, int] = defaultdict(int)
    _accumulate_histogram(label, class_counts)
    return sample_name, int(label.size), class_counts, True


def build_histogram_rows(
    source_folder: Path,
    label_key: str,
    class_prefix: str = DEFAULT_CLASS_PREFIX,
    progress_every: int = 1000,
    max_samples: int | None = None,
    workers: int = 1,
) -> tuple[list[dict[str, object]], list[int], int]:
    rows: list[dict[str, object]] = []
    all_class_ids: set[int] = set()
    skipped_patches = 0
    patch_paths = _list_patch_dirs(source_folder)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        patch_paths = patch_paths[:max_samples]
    if not patch_paths:
        raise FileNotFoundError(f"No patch folders found in {source_folder}")

    workers = max(1, workers)
    if workers == 1:
        for i, (sample_name, total_pixels, class_counts, ok) in enumerate(
            (_read_sample_histogram(sample_path, label_key) for sample_path in (str(p) for p in patch_paths)),
            start=1,
        ):
            if not ok:
                skipped_patches += 1
                continue
            all_class_ids.update(class_counts)
            rows.append(
                {
                    "sample_id": sample_name,
                    "total_pixels": total_pixels,
                    "class_counts": class_counts,
                }
            )
            if progress_every > 0 and i % progress_every == 0:
                print(f"[INFO] Processed {i}/{len(patch_paths)} samples")
    else:
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            for i, (sample_name, total_pixels, class_counts, ok) in enumerate(
                executor.map(_read_sample_histogram, [str(p) for p in patch_paths], [label_key] * len(patch_paths)),
                start=1,
            ):
                if not ok:
                    skipped_patches += 1
                    continue
                all_class_ids.update(class_counts)
                rows.append(
                    {
                        "sample_id": sample_name,
                        "total_pixels": total_pixels,
                        "class_counts": class_counts,
                    }
                )
                if progress_every > 0 and i % progress_every == 0:
                    print(f"[INFO] Processed {i}/{len(patch_paths)} samples")

    if not rows:
        return [], []
    class_ids = sorted(all_class_ids)
    expanded = []
    for row in rows:
        class_counts = row.pop("class_counts")
        expanded_row: dict[str, object] = {
            "sample_id": row["sample_id"],
            "total_pixels": row["total_pixels"],
        }
        for cls in class_ids:
            expanded_row[f"{class_prefix}{cls}"] = class_counts.get(cls, 0)
        expanded.append(expanded_row)

    return expanded, class_ids, skipped_patches


def write_histogram_csv(rows: list[dict[str, object]], class_ids: list[int], class_prefix: str, output: Path) -> None:
    fieldnames = ["sample_id", "total_pixels"] + [f"{class_prefix}{cls}" for cls in class_ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metadata(
    output_csv: Path,
    source_folder: Path,
    dataset: str,
    split: str,
    class_ids: list[int],
    class_prefix: str,
    class_totals: Mapping[str, int],
    metadata: Path | None = None,
) -> None:
    path = metadata or output_csv.with_suffix(".metadata.json")
    payload = {
        "dataset": dataset,
        "source_folder": str(source_folder),
        "split": split,
        "class_prefix": class_prefix,
        "class_ids": class_ids,
        "class_totals": class_totals,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    dataset = args.dataset.lower()
    dataset_path = _resolve_base_path(root_dir, dataset)
    source_folder = dataset_path / ("trainval" if args.split in {"train", "val"} else args.split)
    if not source_folder.exists():
        raise FileNotFoundError(f"Expected split folder at {source_folder}")

    rows, class_ids, skipped_patches = build_histogram_rows(
        source_folder,
        args.label_key,
        args.class_prefix,
        progress_every=args.progress_every,
        max_samples=args.max_samples,
        workers=args.workers,
    )
    output_csv = Path(args.output_csv) if args.output_csv else (
        Path.cwd() / f"{dataset}_{args.split}_sample_histograms.csv"
    )
    write_histogram_csv(rows, class_ids, args.class_prefix, output_csv)

    class_totals: dict[str, int] = {f"{args.class_prefix}{cls}": 0 for cls in class_ids}
    for row in rows:
        for cls in class_ids:
            class_totals[f"{args.class_prefix}{cls}"] += _coerce_int(row[f"{args.class_prefix}{cls}"])

    write_metadata(
        output_csv,
        source_folder,
        dataset,
        args.split,
        class_ids,
        args.class_prefix,
        class_totals,
        Path(args.metadata_json) if args.metadata_json else None,
    )
    print(f"[OK] Saved {len(rows)} sample histograms to {output_csv}")
    if skipped_patches:
        print(f"[WARN] Skipped {skipped_patches} unreadable samples")


if __name__ == "__main__":
    main()
