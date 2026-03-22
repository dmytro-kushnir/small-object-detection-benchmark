#!/usr/bin/env bash
# EXP-002b: resolution sweep (640, 768, 896, 1024) on datasets/processed/test_run — same val GT as EXP-000.
# Prerequisite: optional ./scripts/run_smoke_test.sh (for prepare + baseline); this script prepares if missing.
# If VRAM OOM: EXP002B_BATCH=2 ./scripts/run_exp002b.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${EXP002B_DEVICE:-${SMOKE_DEVICE:-auto}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print(0 if torch.cuda.is_available() else \"cpu\")" 2>/dev/null || echo cpu)"
fi
BATCH="${EXP002B_BATCH:-4}"
echo "== EXP-002b device: $DEVICE (EXP002B_DEVICE or SMOKE_DEVICE); batch: $BATCH (EXP002B_BATCH) =="

mkdir -p experiments/results

_baseline_train_imgsz() {
  python3 -c "
import sys
from pathlib import Path
import yaml
cfg = Path(sys.argv[1])
if not cfg.is_file():
    sys.exit(0)
d = yaml.safe_load(cfg.read_text(encoding='utf-8')) or {}
v = d.get('imgsz', 0)
print(int(v) if v is not None else 0, end='')
" "$ROOT/experiments/yolo/test_run/config.yaml"
}

echo "== Download COCO128 (if needed) =="
python3 scripts/datasets/download_coco128.py

if [[ ! -f "$ROOT/datasets/processed/test_run/dataset.yaml" ]]; then
  echo "== Prepare baseline dataset → datasets/processed/test_run =="
  python3 scripts/datasets/prepare_dataset.py --config-name=exp000_prepare
else
  echo "== Reuse datasets/processed/test_run =="
fi

BASELINE_CFG="$ROOT/experiments/yolo/test_run/config.yaml"
BASELINE_METRICS="$ROOT/experiments/results/test_run_metrics.json"
BASELINE_PRED="$ROOT/experiments/yolo/test_run/predictions_val.json"
BI="$(_baseline_train_imgsz || true)"
REUSE_640=0
if [[ "$BI" == "640" ]] && [[ -f "$BASELINE_METRICS" ]] && [[ -f "$BASELINE_PRED" ]]; then
  REUSE_640=1
  echo "== Will reuse EXP-000 artifacts for imgsz 640 (test_run config imgsz=640) =="
fi

GT_VAL="$ROOT/datasets/processed/test_run/annotations/instances_val.json"
IMG_VAL="$ROOT/datasets/processed/test_run/images/val"
MANIFEST="$ROOT/datasets/processed/test_run/prepare_manifest.json"

for SZ in 640 768 896 1024; do
  NAME="exp002b_imgsz${SZ}"
  YDIR="$ROOT/experiments/yolo/$NAME"
  MET="$ROOT/experiments/results/${NAME}_metrics.json"
  EID="EXP-002b-imgsz${SZ}"

  echo ""
  echo "########## EXP-002b imgsz=${SZ} ##########"

  if [[ "$SZ" == "640" ]] && [[ "$REUSE_640" -eq 1 ]]; then
    cp -f "$BASELINE_METRICS" "$MET"
    echo "Copied baseline metrics to $MET (no train/infer/eval for 640)."
    continue
  fi

  echo "== Train → $YDIR =="
  python3 scripts/train/train_yolo.py --config-name=train/yolo_exp002b \
    "imgsz=$SZ" "name=$NAME" "device=$DEVICE" "batch=$BATCH"

  WEIGHTS="$YDIR/weights/best.pt"
  if [[ ! -f "$WEIGHTS" ]]; then
    WEIGHTS="$YDIR/weights/last.pt"
  fi
  if [[ ! -f "$WEIGHTS" ]]; then
    echo "No weights under $YDIR/weights/" >&2
    exit 1
  fi

  PRED="$YDIR/predictions_val.json"
  echo "== Inference (val) → $PRED =="
  python3 scripts/inference/infer_yolo.py \
    --weights "$WEIGHTS" \
    --source "$IMG_VAL" \
    --coco-gt "$GT_VAL" \
    --out "$PRED" \
    --device "$DEVICE" \
    --imgsz "$SZ"

  echo "== Evaluation → $MET =="
  python3 scripts/evaluation/evaluate.py \
    --gt "$GT_VAL" \
    --pred "$PRED" \
    --out "$MET" \
    --weights "$WEIGHTS" \
    --images-dir "$IMG_VAL" \
    --device "$DEVICE" \
    --experiment-id "$EID" \
    --prepare-manifest "$MANIFEST" \
    --train-config "$YDIR/config.yaml" \
    --imgsz "$SZ"
done

echo ""
echo "== Summarize sweep → experiments/results/exp002b_resolution_sweep.json =="
python3 scripts/evaluation/summarize_resolution_sweep.py \
  --cwd "$ROOT" \
  --out "$ROOT/experiments/results/exp002b_resolution_sweep.json" \
  --recommendation-out "$ROOT/experiments/results/exp002b_recommendation.md" \
  --plots-dir "$ROOT/experiments/results/plots"

echo "EXP-002b finished."
