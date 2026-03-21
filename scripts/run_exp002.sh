#!/usr/bin/env bash
# EXP-002: Same prepared data as EXP-000 (test_run); train at imgsz=1280 → infer → eval → compare vs baseline.
# Prerequisite: ./scripts/run_smoke_test.sh (baseline metrics + datasets/processed/test_run).
# If VRAM OOM at batch 4: EXP002_BATCH=2 ./scripts/run_exp002.sh (document deviation from strict identity).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${EXP002_DEVICE:-${SMOKE_DEVICE:-auto}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print(0 if torch.cuda.is_available() else \"cpu\")" 2>/dev/null || echo cpu)"
fi
BATCH="${EXP002_BATCH:-4}"
echo "== EXP-002 device: $DEVICE (EXP002_DEVICE or SMOKE_DEVICE=auto|0|cpu); batch: $BATCH (EXP002_BATCH overrides) =="

mkdir -p experiments/results

echo "== 1/6 Download COCO128 (if needed) → datasets/raw/test/coco128 =="
python3 scripts/datasets/download_coco128.py

if [[ ! -f "$ROOT/datasets/processed/test_run/dataset.yaml" ]]; then
  echo "== 2/6 Prepare baseline dataset (missing) → datasets/processed/test_run =="
  python3 scripts/datasets/prepare_dataset.py --config-name=exp000_prepare
else
  echo "== 2/6 Skip prepare (reuse datasets/processed/test_run, same as EXP-000) =="
fi

echo "== 3/6 Train YOLO26n, imgsz=1280, 1 epoch → experiments/yolo/test_run_exp002 =="
python3 scripts/train/train_yolo.py --config-name=train/yolo_exp002 \
  "device=$DEVICE" "batch=$BATCH"

# Same path every run: yolo_exp002.yaml sets exist_ok=true so Ultralytics overwrites this dir instead of test_run_exp0022, …
YOLO_RUN="$ROOT/experiments/yolo/test_run_exp002"
WEIGHTS="$YOLO_RUN/weights/best.pt"
if [[ ! -f "$WEIGHTS" ]]; then
  WEIGHTS="$YOLO_RUN/weights/last.pt"
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "No weights under $YOLO_RUN/weights/" >&2
  exit 1
fi

GT_VAL="$ROOT/datasets/processed/test_run/annotations/instances_val.json"
IMG_VAL="$ROOT/datasets/processed/test_run/images/val"
PRED="$YOLO_RUN/predictions_val.json"
METRICS="$ROOT/experiments/results/test_run_exp002_metrics.json"
BASELINE_METRICS="$ROOT/experiments/results/test_run_metrics.json"

echo "== 4/6 Inference (val, same GT/images as EXP-000) → $PRED =="
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
  --experiment-id EXP-002 \
  --prepare-manifest "$ROOT/datasets/processed/test_run/prepare_manifest.json" \
  --train-config "$YOLO_RUN/config.yaml"

EXP002_NOTE="EXP-002 retrains on the same prepared COCO128 as EXP-000 (datasets/processed/test_run); only Ultralytics imgsz is raised (1280 vs 320 in smoke). Val GT and image files are identical to EXP-000. mAP compares the same labels; inference_benchmark reflects higher-resolution predict cost."

echo "== 6/6 Compare vs baseline =="
if [[ -f "$BASELINE_METRICS" ]]; then
  python3 scripts/evaluation/compare_metrics.py \
    --baseline "$BASELINE_METRICS" \
    --compare "$METRICS" \
    --out "$ROOT/experiments/results/exp002_vs_baseline.json" \
    --summary-label "EXP-002 vs baseline" \
    --evaluation-note "$EXP002_NOTE"
else
  echo "Skip compare: baseline not found at $BASELINE_METRICS (run ./scripts/run_smoke_test.sh first)." >&2
fi

echo "EXP-002 finished. Metrics: experiments/results/test_run_exp002_metrics.json"
echo "Figures: VIZ_PRESET=exp002 ./scripts/run_visualization.sh"
