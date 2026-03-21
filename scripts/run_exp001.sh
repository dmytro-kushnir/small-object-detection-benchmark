#!/usr/bin/env bash
# EXP-001: COCO128 with small-object filter → train → infer → eval → compare vs EXP-000 baseline.
# Prerequisite: EXP-000 already run (experiments/results/test_run_metrics.json) for comparison step.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${EXP001_DEVICE:-${SMOKE_DEVICE:-auto}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print(0 if torch.cuda.is_available() else \"cpu\")" 2>/dev/null || echo cpu)"
fi
echo "== EXP-001 device: $DEVICE (EXP001_DEVICE or SMOKE_DEVICE=auto|0|cpu) =="

mkdir -p experiments/results

echo "== 1/6 Download COCO128 (if needed) → datasets/raw/test/coco128 =="
python3 scripts/datasets/download_coco128.py

echo "== 2/6 Prepare filtered dataset → datasets/processed/test_run_exp001 =="
python3 scripts/datasets/prepare_dataset.py --config-name=exp001_prepare

echo "== 3/6 Train YOLO26n, 1 epoch → experiments/yolo/test_run_exp001 =="
python3 scripts/train/train_yolo.py \
  data=datasets/processed/test_run_exp001/dataset.yaml \
  model=yolo26n.pt epochs=1 batch=4 imgsz=320 workers=0 \
  project=experiments/yolo name=test_run_exp001 "device=$DEVICE"

WEIGHTS="$ROOT/experiments/yolo/test_run_exp001/weights/best.pt"
if [[ ! -f "$WEIGHTS" ]]; then
  WEIGHTS="$ROOT/experiments/yolo/test_run_exp001/weights/last.pt"
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "No weights under experiments/yolo/test_run_exp001/weights/" >&2
  exit 1
fi

GT_VAL="$ROOT/datasets/processed/test_run_exp001/annotations/instances_val.json"
IMG_VAL="$ROOT/datasets/processed/test_run_exp001/images/val"
PRED="$ROOT/experiments/yolo/test_run_exp001/predictions_val.json"
METRICS="$ROOT/experiments/results/test_run_exp001_metrics.json"
BASELINE_METRICS="$ROOT/experiments/results/test_run_metrics.json"

echo "== 4/6 Inference (val) → $PRED =="
python3 scripts/inference/infer_yolo.py \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$PRED" \
  --device "$DEVICE"

echo "== 5/6 Evaluation → $METRICS =="
python3 scripts/evaluation/evaluate.py \
  --gt "$GT_VAL" \
  --pred "$PRED" \
  --out "$METRICS" \
  --weights "$WEIGHTS" \
  --images-dir "$IMG_VAL" \
  --device "$DEVICE" \
  --experiment-id EXP-001 \
  --prepare-manifest "$ROOT/datasets/processed/test_run_exp001/prepare_manifest.json" \
  --train-config "$ROOT/experiments/yolo/test_run_exp001/config.yaml"

echo "== 6/6 Compare vs baseline =="
if [[ -f "$BASELINE_METRICS" ]]; then
  python3 scripts/evaluation/compare_metrics.py \
    --baseline "$BASELINE_METRICS" \
    --compare "$METRICS" \
    --out "$ROOT/experiments/results/exp001_vs_baseline.json"
else
  echo "Skip compare: baseline not found at $BASELINE_METRICS (run ./scripts/run_smoke_test.sh first)." >&2
fi

echo "EXP-001 finished. Metrics: experiments/results/test_run_exp001_metrics.json"
echo "Figures: VIZ_PRESET=exp001 ./scripts/run_visualization.sh"
