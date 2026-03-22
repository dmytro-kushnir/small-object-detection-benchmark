#!/usr/bin/env bash
# EXP-A000 full baseline: train 20 epochs → infer → evaluate → relative metrics → viz → summary.
# Outputs under experiments/yolo/ants_expA000_full/, experiments/results/*_full*, visualizations/ants_expA000_full/.
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

mkdir -p "$ROOT/experiments/results" "$ROOT/experiments/visualizations"

echo "== EXP-A000 full train (device=$DEVICE) =="
python3 "$ROOT/scripts/train/train_yolo.py" \
  --config-name=train/yolo_ants_expA000_full \
  "device=$DEVICE" \
  "$@"

YOLO_DIR="$ROOT/experiments/yolo/ants_expA000_full"
WEIGHTS="$YOLO_DIR/weights/best.pt"
[[ -f "$WEIGHTS" ]] || WEIGHTS="$YOLO_DIR/weights/last.pt"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "No weights in $YOLO_DIR/weights/" >&2
  exit 1
fi

PRED="$YOLO_DIR/predictions_val.json"
METRICS="$ROOT/experiments/results/ants_expA000_full_metrics.json"
REL_JSON="$ROOT/experiments/results/ants_expA000_relative_metrics.json"
TC="$YOLO_DIR/config.yaml"
VIZ_OUT="$ROOT/experiments/visualizations/ants_expA000_full"
SMOKE_METRICS="$ROOT/experiments/results/ants_expA000_smoke_metrics.json"

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
  --experiment-id "EXP-A000-full" \
  --prepare-manifest "$MANIFEST" \
  --train-config "$TC" \
  --imgsz 640

echo "== Relative size metrics → $REL_JSON =="
python3 "$ROOT/scripts/evaluation/ants_relative_size_metrics.py" \
  --coco-gt "$GT_VAL" \
  --pred "$PRED" \
  --out "$REL_JSON" \
  --score-thr 0.25

# Default cap 250 images; set ANTS_VIZ_MAX_IMAGES=all for no cap, or e.g. 1073 for full val.
VIZ_ARGS=()
if [[ "${ANTS_VIZ_MAX_IMAGES:-}" == "all" ]]; then
  :
elif [[ -n "${ANTS_VIZ_MAX_IMAGES:-}" ]]; then
  VIZ_ARGS+=(--max-images "$ANTS_VIZ_MAX_IMAGES")
else
  VIZ_ARGS+=(--max-images 250)
fi

echo "== Viz → $VIZ_OUT (max-images: ${ANTS_VIZ_MAX_IMAGES:-250 default}) =="
python3 "$ROOT/scripts/visualization/viz_coco_overlays.py" \
  --pred "$PRED" \
  --gt "$GT_VAL" \
  --images-dir "$IMG_VAL" \
  --out-dir "$VIZ_OUT" \
  "${VIZ_ARGS[@]}"

SUMMARY_OUT="$ROOT/experiments/results/ants_expA000_full_summary.md"
SMOKE_ARG=()
if [[ -f "$SMOKE_METRICS" ]]; then
  SMOKE_ARG=(--metrics-smoke "$SMOKE_METRICS")
fi

echo "== Full summary → $SUMMARY_OUT =="
python3 "$ROOT/scripts/evaluation/write_ants_expA000_full_summary.py" \
  --metrics-full "$METRICS" \
  "${SMOKE_ARG[@]}" \
  --relative "$REL_JSON" \
  --train-config "$TC" \
  --analysis "$ROOT/datasets/ants_yolo/analysis.json" \
  --out "$SUMMARY_OUT"

echo "EXP-A000 full finished. Metrics: $METRICS  Summary: $SUMMARY_OUT"
