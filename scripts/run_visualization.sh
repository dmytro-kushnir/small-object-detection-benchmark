#!/usr/bin/env bash
# GT/pred overlays from existing predictions + COCO (no training / no evaluation).
# Preset: VIZ_PRESET=exp000|exp001|exp002, or optional first argument (then passed-through args follow).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Default predictions path for a YOLO run name. If RUN_SUB contains _exp (e.g. test_run_exp001),
# Ultralytics may create RUN_SUB2, RUN_SUB3, … — pick newest predictions_val.json by mtime across experiments/yolo/RUN_SUB*.
_default_yolo_predictions_json() {
  local sub="$1"
  local def="$ROOT/experiments/yolo/$sub/predictions_val.json"
  if [[ "$sub" != *_exp* ]]; then
    echo "$def"
    return
  fi
  local best="" ts=0 d f m
  for d in "$ROOT/experiments/yolo/${sub}"*; do
    [[ -d "$d" ]] || continue
    f="$d/predictions_val.json"
    [[ -f "$f" ]] || continue
    m="$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || echo 0)"
    if [[ "$m" -gt "$ts" ]]; then
      ts="$m"
      best="$f"
    fi
  done
  echo "${best:-$def}"
}

VIZ_PRESET="${VIZ_PRESET:-exp000}"
if [[ "${1:-}" == exp000 ]] || [[ "${1:-}" == exp001 ]] || [[ "${1:-}" == exp002 ]]; then
  VIZ_PRESET="$1"
  shift
fi

case "$VIZ_PRESET" in
  exp000)
    RUN_SUB="test_run"
    DS_SUB="test_run"
    ;;
  exp001)
    RUN_SUB="test_run_exp001"
    DS_SUB="test_run_exp001"
    ;;
  exp002)
    RUN_SUB="test_run_exp002"
    DS_SUB="test_run"
    ;;
  *)
    echo "Unknown VIZ_PRESET=$VIZ_PRESET (use exp000, exp001, or exp002)" >&2
    exit 1
    ;;
esac

if [[ -z "${PRED-}" ]]; then
  PRED="$(_default_yolo_predictions_json "$RUN_SUB")"
fi
PRED="${PRED:-$ROOT/experiments/yolo/${RUN_SUB}/predictions_val.json}"
GT="${GT:-$ROOT/datasets/processed/${DS_SUB}/annotations/instances_val.json}"
IMG="${IMAGES_DIR:-$ROOT/datasets/processed/${DS_SUB}/images/val}"
OUT="${VIZ_OUT:-$ROOT/experiments/visualizations/${RUN_SUB}}"

if [[ -f "$ROOT/datasets/processed/${DS_SUB}/annotations/instances_train.json" ]] \
   && [[ -d "$ROOT/datasets/processed/${DS_SUB}/images/train" ]]; then
  DS_GT="${DATASET_GT:-$ROOT/datasets/processed/${DS_SUB}/annotations/instances_train.json}"
  DS_IM="${DATASET_IMAGES:-$ROOT/datasets/processed/${DS_SUB}/images/train}"
else
  DS_GT="${DATASET_GT:-$GT}"
  DS_IM="${DATASET_IMAGES:-$IMG}"
fi

python3 scripts/visualization/viz_coco_overlays.py \
  --pred "$PRED" \
  --gt "$GT" \
  --images-dir "$IMG" \
  --out-dir "$OUT" \
  --dataset-gt "$DS_GT" \
  --dataset-images-dir "$DS_IM" \
  "$@"

echo "Visualizations under: $OUT"
