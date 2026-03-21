#!/usr/bin/env bash
# GT/pred overlays from existing predictions + COCO (no training / no evaluation).
# Preset: VIZ_PRESET=exp000|exp001, or optional first argument exp000|exp001 (then passed-through args follow).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VIZ_PRESET="${VIZ_PRESET:-exp000}"
if [[ "${1:-}" == exp000 ]] || [[ "${1:-}" == exp001 ]]; then
  VIZ_PRESET="$1"
  shift
fi

case "$VIZ_PRESET" in
  exp000)
    RUN_SUB="test_run"
    ;;
  exp001)
    RUN_SUB="test_run_exp001"
    ;;
  *)
    echo "Unknown VIZ_PRESET=$VIZ_PRESET (use exp000 or exp001)" >&2
    exit 1
    ;;
esac

PRED="${PRED:-$ROOT/experiments/yolo/${RUN_SUB}/predictions_val.json}"
GT="${GT:-$ROOT/datasets/processed/${RUN_SUB}/annotations/instances_val.json}"
IMG="${IMAGES_DIR:-$ROOT/datasets/processed/${RUN_SUB}/images/val}"
OUT="${VIZ_OUT:-$ROOT/experiments/visualizations/${RUN_SUB}}"

if [[ -f "$ROOT/datasets/processed/${RUN_SUB}/annotations/instances_train.json" ]] \
   && [[ -d "$ROOT/datasets/processed/${RUN_SUB}/images/train" ]]; then
  DS_GT="${DATASET_GT:-$ROOT/datasets/processed/${RUN_SUB}/annotations/instances_train.json}"
  DS_IM="${DATASET_IMAGES:-$ROOT/datasets/processed/${RUN_SUB}/images/train}"
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
