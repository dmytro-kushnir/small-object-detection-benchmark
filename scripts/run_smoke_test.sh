#!/usr/bin/env bash
# EXP-000 baseline: download COCO128 → prepare → train YOLO26n (1 ep) → infer → pycocotools eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${SMOKE_DEVICE:-auto}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print(0 if torch.cuda.is_available() else \"cpu\")" 2>/dev/null || echo cpu)"
fi
echo "== EXP-000 device: $DEVICE (SMOKE_DEVICE=auto|0|cpu) =="

mkdir -p experiments/results

echo "== 1/5 Download COCO128 → datasets/raw/test/coco128 =="
python3 scripts/datasets/download_coco128.py

echo "== 2/5 Prepare dataset → datasets/processed/test_run =="
python3 scripts/datasets/prepare_dataset.py --config-name=exp000_prepare

echo "== 3/5 Train YOLO26n, 1 epoch → experiments/yolo/test_run =="
python3 scripts/train/train_yolo.py \
  data=datasets/processed/test_run/dataset.yaml \
  model=yolo26n.pt epochs=1 batch=4 imgsz=320 workers=0 \
  project=experiments/yolo name=test_run "device=$DEVICE"

WEIGHTS="$ROOT/experiments/yolo/test_run/weights/best.pt"
if [[ ! -f "$WEIGHTS" ]]; then
  WEIGHTS="$ROOT/experiments/yolo/test_run/weights/last.pt"
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "No weights under experiments/yolo/test_run/weights/" >&2
  exit 1
fi

GT_VAL="$ROOT/datasets/processed/test_run/annotations/instances_val.json"
IMG_VAL="$ROOT/datasets/processed/test_run/images/val"
PRED="$ROOT/experiments/yolo/test_run/predictions_val.json"
METRICS="$ROOT/experiments/results/test_run_metrics.json"

echo "== 4/5 Inference (val images) → $PRED =="
python3 scripts/inference/infer_yolo.py \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$PRED" \
  --device "$DEVICE"

echo "== 5/5 Evaluation (pycocotools + FPS) → $METRICS =="
python3 scripts/evaluation/evaluate.py \
  --gt "$GT_VAL" \
  --pred "$PRED" \
  --out "$METRICS" \
  --weights "$WEIGHTS" \
  --images-dir "$IMG_VAL" \
  --device "$DEVICE" \
  --prepare-manifest "$ROOT/datasets/processed/test_run/prepare_manifest.json" \
  --train-config "$ROOT/experiments/yolo/test_run/config.yaml"

echo "EXP-000 finished. Metrics: experiments/results/test_run_metrics.json"
