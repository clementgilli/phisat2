# Segmentation split stratification scripts

These scripts are intentionally scoped to **segmentation datasets**.

## 1) Build per-sample class histograms

```bash
python scripts/segmentation/build_sample_histograms.py \
  --dataset lulc \
  --root-dir /path/to/data \
  --split train \
  --output-csv /tmp/lulc_train_sample_hist.csv
```

- `--dataset` follows the existing dataset naming used by the project (`lc`, `lulc`, `marine`, `floods`, `burned`, ...).
- The script writes one row per sample and one `class_<id>` column per class found in the masks.

## 2) Select a stratified subset from the histogram

```bash
python scripts/segmentation/stratify_samples.py \
  --hist-csv /tmp/lulc_train_sample_hist.csv \
  --train-size 50 \
  --strategy global \
  --selected-csv /tmp/lulc_train_50_global.csv \
  --summary-json /tmp/lulc_train_50_global.json

python scripts/segmentation/stratify_samples.py \
  --hist-csv /tmp/lulc_train_sample_hist.csv \
  --train-size 5000 \
  --strategy balanced \
  --selected-csv /tmp/lulc_train_5000_balanced.csv \
  --summary-json /tmp/lulc_train_5000_balanced.json
```

`full` is supported for baseline:

```bash
python scripts/segmentation/stratify_samples.py \
  --hist-csv /tmp/lulc_train_sample_hist.csv \
  --train-size full \
  --strategy global
```

### Outputs

- `<strategy>.N.selected.csv`: selected sample ids in selection order (with class counts).
- `<strategy>.N.summary.json`: full and selected class counts/fractions and `l1_distance`.
- optional `--summary-csv`: per-class compact CSV for quick diffing.
