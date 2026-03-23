#!/usr/bin/env bash
# Run ANTS val inference + evaluate.py at four pipeline stages (diagnostic; 4× infer cost).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${ANTS_DEVICE:-${SMOKE_DEVICE:-auto}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print(0 if torch.cuda.is_available() else \"cpu\")" 2>/dev/null || echo cpu)"
fi

ANTS_CFG="${EXP_A004_ANT_CONFIG:-$ROOT/configs/expA004_ants_v1.yaml}"
GT_VAL="$ROOT/datasets/ants_yolo/annotations/instances_val.json"
IMG_VAL="$ROOT/datasets/ants_yolo/images/val"
MANIFEST="$ROOT/datasets/ants_yolo/prepare_manifest.json"
YOLO_768="$ROOT/experiments/yolo/ants_expA002b_imgsz768"
WEIGHTS="$YOLO_768/weights/best.pt"
[[ -f "$WEIGHTS" ]] || WEIGHTS="$YOLO_768/weights/last.pt"
TC_768="$YOLO_768/config.yaml"
SCRATCH="$ROOT/experiments/yolo/ants_expA004_staged_scratch"
RES="$ROOT/experiments/results"

mkdir -p "$SCRATCH" "$RES"

run_eval() {
  local pred="$1"
  local out="$2"
  local eid="$3"
  python3 "$ROOT/scripts/evaluation/evaluate.py" \
    --gt "$GT_VAL" \
    --pred "$pred" \
    --out "$out" \
    --weights "$WEIGHTS" \
    --images-dir "$IMG_VAL" \
    --device "$DEVICE" \
    --experiment-id "$eid" \
    --prepare-manifest "$MANIFEST" \
    --train-config "$TC_768" \
    --skip-inference-benchmark
}

echo "== Stage 1: pipeline_mode=stage1_only → $SCRATCH/preds_stage1_only.json =="
python3 "$ROOT/scripts/inference/infer_ants_v1.py" \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$SCRATCH/preds_stage1_only.json" \
  --config "$ANTS_CFG" \
  --device "$DEVICE" \
  --pipeline-mode stage1_only \
  --rois-out "$SCRATCH/rois_stage1_only.json" \
  --stage1-out "$SCRATCH/stage1_mirror.json" \
  --no-progress
run_eval "$SCRATCH/preds_stage1_only.json" "$RES/ants_expA004_staged_stage1_only_metrics.json" "EXP-A004-staged-stage1-only"

echo "== Stage 2: parity-baseline (dense ROIs off, merged path) → $SCRATCH/preds_parity.json =="
python3 "$ROOT/scripts/inference/infer_ants_v1.py" \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$SCRATCH/preds_parity.json" \
  --config "$ANTS_CFG" \
  --device "$DEVICE" \
  --parity-baseline \
  --rois-out "$SCRATCH/rois_parity.json" \
  --stage1-out "$SCRATCH/stage1_parity.json" \
  --no-progress
run_eval "$SCRATCH/preds_parity.json" "$RES/ants_expA004_staged_parity_metrics.json" "EXP-A004-staged-parity"

echo "== Stage 3: union_refined → $SCRATCH/preds_union.json =="
python3 "$ROOT/scripts/inference/infer_ants_v1.py" \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$SCRATCH/preds_union.json" \
  --config "$ANTS_CFG" \
  --device "$DEVICE" \
  --pipeline-mode union_refined \
  --rois-out "$SCRATCH/rois_union.json" \
  --stage1-out "$SCRATCH/stage1_union.json" \
  --no-progress
run_eval "$SCRATCH/preds_union.json" "$RES/ants_expA004_staged_union_metrics.json" "EXP-A004-staged-union"

echo "== Stage 4: full merged (default) → $SCRATCH/preds_full.json =="
python3 "$ROOT/scripts/inference/infer_ants_v1.py" \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$SCRATCH/preds_full.json" \
  --config "$ANTS_CFG" \
  --device "$DEVICE" \
  --rois-out "$SCRATCH/rois_full.json" \
  --stage1-out "$SCRATCH/stage1_full.json" \
  --no-progress
run_eval "$SCRATCH/preds_full.json" "$RES/ants_expA004_staged_full_metrics.json" "EXP-A004-staged-full"

echo "Staged metrics under $RES/ants_expA004_staged_*_metrics.json"
