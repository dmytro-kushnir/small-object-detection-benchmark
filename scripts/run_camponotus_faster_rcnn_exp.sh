#!/usr/bin/env bash
# EXP-CAMPO-FRCNN: Camponotus track-majority Faster R-CNN — train → infer → bench → evaluate → compare vs YOLO.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="${EXP_FRCNN_CONFIG:-$ROOT/configs/camponotus_faster_rcnn_trackidmajor_896.yaml}"

COCO_ROOT="${EXP_FRCNN_COCO_ROOT:-$ROOT/datasets/camponotus_coco/camponotus_full_export_unique_trackidmajor}"
YOLO_IMG_ROOT="${EXP_FRCNN_YOLO_IMG_ROOT:-$ROOT/datasets/camponotus_yolo/camponotus_full_export_unique_trackidmajor}"

GT_TRAIN="${EXP_FRCNN_GT_TRAIN:-$COCO_ROOT/annotations/instances_train.json}"
GT_VAL="${EXP_FRCNN_GT_VAL:-$COCO_ROOT/annotations/instances_val.json}"
GT_TEST="${EXP_FRCNN_GT_TEST:-$COCO_ROOT/annotations/instances_test.json}"
IMG_TRAIN="${EXP_FRCNN_IMG_TRAIN:-$YOLO_IMG_ROOT/images/train}"
IMG_VAL="${EXP_FRCNN_IMG_VAL:-$YOLO_IMG_ROOT/images/val}"
IMG_TEST="${EXP_FRCNN_IMG_TEST:-$YOLO_IMG_ROOT/images/test}"

OUT_DIR="${EXP_FRCNN_OUT_DIR:-$ROOT/experiments/faster_rcnn/camponotus_faster_rcnn_trackidmajor_896}"
PRED_VAL="${EXP_FRCNN_PRED_VAL:-$OUT_DIR/predictions_val.json}"
PRED_TEST="${EXP_FRCNN_PRED_TEST:-$OUT_DIR/predictions_test.json}"
BENCH_VAL="${EXP_FRCNN_BENCH_VAL:-$OUT_DIR/inference_benchmark_val.json}"
BENCH_TEST="${EXP_FRCNN_BENCH_TEST:-$OUT_DIR/inference_benchmark_test.json}"
MET_VAL="${EXP_FRCNN_MET_VAL:-$ROOT/experiments/results/camponotus_faster_rcnn_trackidmajor_896_metrics_val.json}"
MET_TEST="${EXP_FRCNN_MET_TEST:-$ROOT/experiments/results/camponotus_faster_rcnn_trackidmajor_896_metrics_test.json}"
CMP_VAL="${EXP_FRCNN_CMP_VAL:-$ROOT/experiments/results/camponotus_faster_rcnn_vs_yolo8962_val.json}"
CMP_TEST="${EXP_FRCNN_CMP_TEST:-$ROOT/experiments/results/camponotus_faster_rcnn_vs_yolo8962_test.json}"
YOLO_MET_VAL="${EXP_FRCNN_YOLO_MET_VAL:-$ROOT/experiments/results/camponotus_trackidmajor_full_8962_metrics_val.json}"
YOLO_MET_TEST="${EXP_FRCNN_YOLO_MET_TEST:-$ROOT/experiments/results/camponotus_trackidmajor_full_8962_metrics_test.json}"
EXP_ID_VAL="${EXP_FRCNN_EXP_ID_VAL:-EXP-CAMPO-FRCNN-VAL}"
EXP_ID_TEST="${EXP_FRCNN_EXP_ID_TEST:-EXP-CAMPO-FRCNN-TEST}"

WEIGHTS="$OUT_DIR/weights/best.pth"
TRAIN_CFG_OUT="$OUT_DIR/config.yaml"

DEVICE="${EXP_FRCNN_DEVICE:-${CAMPO_DEVICE:-${SMOKE_DEVICE:-auto}}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo cpu)"
fi

MIN_SIZE="$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1],encoding='utf-8')).get('min_size',896))" "$CFG" 2>/dev/null || echo 896)"
MAX_SIZE="$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1],encoding='utf-8')).get('max_size',1333))" "$CFG" 2>/dev/null || echo 1333)"
CONF_THR="$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1],encoding='utf-8')).get('conf_threshold',0.25))" "$CFG" 2>/dev/null || echo 0.25)"

echo "== EXP-CAMPO-FRCNN config: $CFG =="
echo "== device: $DEVICE =="

for f in "$GT_VAL" "$GT_TEST" "$IMG_VAL" "$IMG_TEST"; do
  if [[ ! -e "$f" ]]; then
    echo "Missing required path: $f" >&2
    echo "Prepare Camponotus bundle first (see scripts/run_camponotus_cvat_bundle_prepare_train.sh)." >&2
    exit 1
  fi
done

MAX_ARGS=()
TRAIN_EXTRA=()
if [[ -n "${EXP_FRCNN_MAX_IMAGES:-}" ]]; then
  MAX_ARGS+=(--max-images "$EXP_FRCNN_MAX_IMAGES")
  TRAIN_EXTRA+=(--max-train-images "$EXP_FRCNN_MAX_IMAGES" --max-val-images "$EXP_FRCNN_MAX_IMAGES")
fi
if [[ -n "${EXP_FRCNN_EPOCHS:-}" ]]; then
  TRAIN_EXTRA+=(--epochs "$EXP_FRCNN_EPOCHS")
fi

if [[ "${EXP_FRCNN_SKIP_TRAIN:-0}" != "1" ]]; then
  echo "== Train Faster R-CNN → $OUT_DIR =="
  python3 "$ROOT/scripts/train/train_faster_rcnn.py" \
    --config "$CFG" \
    "${TRAIN_EXTRA[@]}"
else
  echo "== Skip train (EXP_FRCNN_SKIP_TRAIN=1) =="
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing weights: $WEIGHTS" >&2
  exit 1
fi

_run_split() {
  local split_name="$1"
  local gt="$2"
  local imgdir="$3"
  local pred="$4"
  local bench="$5"
  local met="$6"
  local exp_id="$7"

  echo "== Infer $split_name → $pred =="
  python3 "$ROOT/scripts/inference/infer_faster_rcnn.py" \
    --weights "$WEIGHTS" \
    --source "$imgdir" \
    --coco-gt "$gt" \
    --out "$pred" \
    --conf "$CONF_THR" \
    --min-size "$MIN_SIZE" \
    --max-size "$MAX_SIZE" \
    --device "$DEVICE" \
    "${MAX_ARGS[@]}"

  echo "== Bench $split_name → $bench =="
  python3 "$ROOT/scripts/evaluation/bench_faster_rcnn.py" \
    --weights "$WEIGHTS" \
    --source "$imgdir" \
    --coco-gt "$gt" \
    --conf "$CONF_THR" \
    --min-size "$MIN_SIZE" \
    --max-size "$MAX_SIZE" \
    --device "$DEVICE" \
    --out "$bench" \
    --config "$CFG" \
    "${MAX_ARGS[@]}"

  mkdir -p "$ROOT/experiments/results"
  echo "== Evaluate $split_name → $met =="
  python3 "$ROOT/scripts/evaluation/evaluate.py" \
    --gt "$gt" \
    --pred "$pred" \
    --weights "$WEIGHTS" \
    --images-dir "$imgdir" \
    --out "$met" \
    --experiment-id "$exp_id" \
    --train-config "$TRAIN_CFG_OUT" \
    --device "$DEVICE" \
    --inference-benchmark-json "$bench"
}

_run_split "val" "$GT_VAL" "$IMG_VAL" "$PRED_VAL" "$BENCH_VAL" "$MET_VAL" "$EXP_ID_VAL"
_run_split "test" "$GT_TEST" "$IMG_TEST" "$PRED_TEST" "$BENCH_TEST" "$MET_TEST" "$EXP_ID_TEST"

echo "== Compare val vs YOLO8962 → $CMP_VAL =="
if [[ -f "$YOLO_MET_VAL" ]]; then
  python3 "$ROOT/scripts/evaluation/compare_metrics.py" \
    --baseline "$YOLO_MET_VAL" \
    --compare "$MET_VAL" \
    --out "$CMP_VAL" \
    --evaluation-note "Faster R-CNN R50-FPN vs YOLO26n track-majority @896; resize policy differs (torchvision min_size vs YOLO letterbox)."
else
  echo "(Skip val compare: baseline not found at $YOLO_MET_VAL)"
fi

echo "== Compare test vs YOLO8962 → $CMP_TEST =="
if [[ -f "$YOLO_MET_TEST" ]]; then
  python3 "$ROOT/scripts/evaluation/compare_metrics.py" \
    --baseline "$YOLO_MET_TEST" \
    --compare "$MET_TEST" \
    --out "$CMP_TEST" \
    --evaluation-note "Faster R-CNN R50-FPN vs YOLO26n track-majority @896; resize policy differs (torchvision min_size vs YOLO letterbox)."
else
  echo "(Skip test compare: baseline not found at $YOLO_MET_TEST)"
fi

echo "EXP-CAMPO-FRCNN finished."
