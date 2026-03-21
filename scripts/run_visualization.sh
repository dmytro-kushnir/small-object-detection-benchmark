#!/usr/bin/env bash
# EXP-000 visualizations from existing predictions + COCO GT (no training / no evaluation).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRED="${PRED:-$ROOT/experiments/yolo/test_run/predictions_val.json}"
GT="${GT:-$ROOT/datasets/processed/test_run/annotations/instances_val.json}"
IMG="${IMAGES_DIR:-$ROOT/datasets/processed/test_run/images/val}"
OUT="${VIZ_OUT:-$ROOT/experiments/visualizations/test_run}"

if [[ -f "$ROOT/datasets/processed/test_run/annotations/instances_train.json" ]] \
   && [[ -d "$ROOT/datasets/processed/test_run/images/train" ]]; then
  DS_GT="${DATASET_GT:-$ROOT/datasets/processed/test_run/annotations/instances_train.json}"
  DS_IM="${DATASET_IMAGES:-$ROOT/datasets/processed/test_run/images/train}"
else
  DS_GT="${DATASET_GT:-$GT}"
  DS_IM="${DATASET_IMAGES:-$IMG}"
fi

python3 scripts/visualization/viz_exp000.py \
  --pred "$PRED" \
  --gt "$GT" \
  --images-dir "$IMG" \
  --out-dir "$OUT" \
  --dataset-gt "$DS_GT" \
  --dataset-images-dir "$DS_IM" \
  "$@"

echo "Visualizations under: $OUT"
