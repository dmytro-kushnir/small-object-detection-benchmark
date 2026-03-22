#!/usr/bin/env bash
# EXP-A000 smoke: train 1 epoch → val infer → evaluate (requires prepared datasets/ants_yolo).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${ANTS_DEVICE:-${SMOKE_DEVICE:-auto}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print(0 if torch.cuda.is_available() else \"cpu\")" 2>/dev/null || echo cpu)"
fi

DATA_YAML="$ROOT/datasets/ants_yolo/dataset.yaml"
GT_VAL="$ROOT/datasets/ants_yolo/annotations/instances_val.json"
IMG_VAL="$ROOT/datasets/ants_yolo/images/val"
MANIFEST="$ROOT/datasets/ants_yolo/prepare_manifest.json"

for f in "$DATA_YAML" "$GT_VAL" "$IMG_VAL"; do
  if [[ ! -e "$f" ]]; then
    echo "Missing $f — run ./scripts/run_ants_prepare.sh first (set ANTS_DATASET_ROOT)." >&2
    exit 1
  fi
done

mkdir -p "$ROOT/experiments/results"

echo "== EXP-A000 smoke train (device=$DEVICE) =="
python3 "$ROOT/scripts/train/train_yolo.py" \
  --config-name=train/yolo_ants_expA000_smoke \
  "device=$DEVICE" \
  "$@"

YOLO_DIR="$ROOT/experiments/yolo/ants_expA000_smoke"
WEIGHTS="$YOLO_DIR/weights/best.pt"
[[ -f "$WEIGHTS" ]] || WEIGHTS="$YOLO_DIR/weights/last.pt"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "No weights in $YOLO_DIR/weights/" >&2
  exit 1
fi

PRED="$YOLO_DIR/predictions_val.json"
METRICS="$ROOT/experiments/results/ants_expA000_smoke_metrics.json"
TC="$YOLO_DIR/config.yaml"

echo "== Inference val → $PRED =="
python3 "$ROOT/scripts/inference/infer_yolo.py" \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$PRED" \
  --device "$DEVICE" \
  --imgsz 640

echo "== Evaluate → $METRICS =="
python3 "$ROOT/scripts/evaluation/evaluate.py" \
  --gt "$GT_VAL" \
  --pred "$PRED" \
  --out "$METRICS" \
  --weights "$WEIGHTS" \
  --images-dir "$IMG_VAL" \
  --device "$DEVICE" \
  --experiment-id "EXP-A000-smoke" \
  --prepare-manifest "$MANIFEST" \
  --train-config "$TC" \
  --imgsz 640

echo "EXP-A000 smoke finished. Metrics: $METRICS"
