#!/usr/bin/env bash
# EXP-A005: RF-DETR baseline on ants — prepare COCO → train → infer → bench → eval → compare → viz → summary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Reduce noisy third-party warnings and improve CUDA allocator behavior.
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG="${EXP_A005_CONFIG:-$ROOT/configs/expA005_ants_rfdetr.yaml}"

DATASET_DIR="${EXP_A005_DATASET_DIR:-$ROOT/datasets/ants_coco}"
OUT_DIR="${EXP_A005_OUT_DIR:-$ROOT/experiments/rfdetr/ants_expA005}"
GT_VAL="${EXP_A005_GT_VAL:-$DATASET_DIR/valid/_annotations.coco.json}"
IMG_VAL="${EXP_A005_IMG_VAL:-$DATASET_DIR/valid}"
PRED_OUT="${EXP_A005_PRED:-$OUT_DIR/predictions_val.json}"
BENCH_JSON="${EXP_A005_BENCH:-$OUT_DIR/inference_benchmark.json}"
MET_OUT="${EXP_A005_METRICS_OUT:-$ROOT/experiments/results/ants_expA005_rfdetr_metrics.json}"
CMP_OUT="${EXP_A005_COMPARE_OUT:-$ROOT/experiments/results/ants_expA005_rfdetr_vs_yolo.json}"
VIZ_DIR="${EXP_A005_VIZ_DIR:-$ROOT/experiments/visualizations/ants_expA005_rfdetr}"
SUMMARY_MD="${EXP_A005_SUMMARY_OUT:-$ROOT/experiments/results/ants_expA005_rfdetr_summary.md}"
YOLO_PRED="${EXP_A005_YOLO_PRED:-$ROOT/experiments/yolo/ants_expA002b_imgsz768/predictions_val.json}"
YOLO_MET="${EXP_A005_YOLO_METRICS:-$ROOT/experiments/results/ants_expA002b_imgsz768_metrics.json}"

MODEL_CLASS="$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['model_class'])" 2>/dev/null || echo RFDETRSmall)"
CONF_THR="$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['conf_threshold'])" 2>/dev/null || echo 0.25)"

WEIGHTS="$OUT_DIR/weights/best.pth"
TRAIN_CFG_OUT="$OUT_DIR/config.yaml"
MANIFEST="$DATASET_DIR/ants_coco_manifest.json"

DEVICE="${RFDETR_DEVICE:-${ANTS_DEVICE:-${SMOKE_DEVICE:-auto}}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo cpu)"
fi

EXP_ID="${EXP_A005_EXPERIMENT_ID:-EXP-A005-ANTS-RFDETR}"

echo "== EXP-A005 config: $CFG =="
echo "== device: $DEVICE =="

if [[ ! -f "$DATASET_DIR/valid/_annotations.coco.json" ]]; then
  echo "== Prepare ants_coco for RF-DETR =="
  python3 "$ROOT/scripts/datasets/prepare_ants_coco_rfdetr.py" \
    --config "$ROOT/configs/datasets/ants_coco_rfdetr.yaml"
fi

if [[ "${EXP_A005_SKIP_TRAIN:-0}" != "1" ]]; then
  echo "== Train RF-DETR → $OUT_DIR =="
  python3 "$ROOT/scripts/train/train_rfdetr_ants.py" --config "$CFG"
else
  echo "== Skip train (EXP_A005_SKIP_TRAIN=1) =="
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing weights: $WEIGHTS (train or copy checkpoint here)" >&2
  exit 1
fi

for f in "$GT_VAL" "$IMG_VAL" "$YOLO_MET"; do
  if [[ ! -e "$f" ]]; then
    echo "Missing: $f" >&2
    exit 1
  fi
done

echo "== Infer val → $PRED_OUT =="
MAX_ARGS=()
if [[ -n "${EXP_A005_MAX_IMAGES:-}" ]]; then
  MAX_ARGS+=(--max-images "$EXP_A005_MAX_IMAGES")
fi
python3 "$ROOT/scripts/inference/infer_rfdetr.py" \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$PRED_OUT" \
  --model-class "$MODEL_CLASS" \
  --conf "$CONF_THR" \
  --device "$DEVICE" \
  "${MAX_ARGS[@]}"

echo "== Bench RF-DETR → $BENCH_JSON =="
BMAX=()
if [[ -n "${EXP_A005_MAX_IMAGES:-}" ]]; then
  BMAX+=(--max-images "$EXP_A005_MAX_IMAGES")
fi
python3 "$ROOT/scripts/evaluation/bench_rfdetr.py" \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --model-class "$MODEL_CLASS" \
  --conf "$CONF_THR" \
  --device "$DEVICE" \
  --out "$BENCH_JSON" \
  --config "$CFG" \
  "${BMAX[@]}"

mkdir -p "$ROOT/experiments/results" "$ROOT/experiments/visualizations"

echo "== Evaluate → $MET_OUT =="
EVAL_EXTRA=()
if [[ -f "$MANIFEST" ]]; then
  EVAL_EXTRA+=(--prepare-manifest "$MANIFEST")
fi
python3 "$ROOT/scripts/evaluation/evaluate.py" \
  --gt "$GT_VAL" \
  --pred "$PRED_OUT" \
  --weights "$WEIGHTS" \
  --images-dir "$IMG_VAL" \
  --out "$MET_OUT" \
  --experiment-id "$EXP_ID" \
  --train-config "$TRAIN_CFG_OUT" \
  "${EVAL_EXTRA[@]}" \
  --device "$DEVICE" \
  --inference-benchmark-json "$BENCH_JSON"

echo "== Compare vs YOLO768 → $CMP_OUT =="
python3 "$ROOT/scripts/evaluation/compare_ants_expA005.py" \
  --baseline "$YOLO_MET" \
  --compare "$MET_OUT" \
  --out "$CMP_OUT"

if [[ -f "$YOLO_PRED" ]]; then
  echo "== Viz → $VIZ_DIR =="
  mkdir -p "$VIZ_DIR"
  VMAX=(--max-images "${EXP_A005_VIZ_MAX_IMAGES:-250}")
  python3 "$ROOT/scripts/visualization/viz_ants_expA005_comparisons.py" \
    --gt "$GT_VAL" \
    --images-dir "$IMG_VAL" \
    --pred-yolo "$YOLO_PRED" \
    --pred-rfdetr "$PRED_OUT" \
    --out-dir "$VIZ_DIR" \
    "${VMAX[@]}"
else
  echo "(Skip viz: YOLO preds not found at $YOLO_PRED)"
fi

echo "== Summary → $SUMMARY_MD =="
python3 "$ROOT/scripts/evaluation/write_ants_expA005_summary.py" \
  --compare "$CMP_OUT" \
  --exp-config "$CFG" \
  --out "$SUMMARY_MD"

echo "EXP-A005 finished."
