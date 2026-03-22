#!/usr/bin/env bash
# Build datasets/ants_yolo from MOT Ant_dataset (configurable path).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${ANTS_DATASET_ROOT:-}" ]]; then
  echo "Set ANTS_DATASET_ROOT to the directory containing MOT sequences (…/gt/gt.txt + img1/)." >&2
  exit 1
fi

echo "== Prepare ants YOLO from: $ANTS_DATASET_ROOT =="
python3 "$ROOT/scripts/datasets/prepare_ants_mot.py" "dataset_root=$ANTS_DATASET_ROOT" "$@"

echo "== GT sample visualizations =="
python3 "$ROOT/scripts/visualization/viz_ant_gt_samples.py" \
  --dataset-yaml "$ROOT/datasets/ants_yolo/dataset.yaml" \
  --split val \
  --n 18 \
  --out-dir "$ROOT/experiments/visualizations/ants_dataset"

echo "Done. Val COCO GT: $ROOT/datasets/ants_yolo/annotations/instances_val.json"
