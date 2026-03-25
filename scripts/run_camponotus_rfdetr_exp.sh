#!/usr/bin/env bash
# EXP-CAMPO-RFDETR: Camponotus two-class RF-DETR — prepare Roboflow COCO → train → infer/bench/eval val+test → compare vs YOLO.
# Paths default to configs/expCAMPO_rfdetr.yaml; override with env vars (see below).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG="${EXP_CAMPO_RFDETR_CONFIG:-$ROOT/configs/expCAMPO_rfdetr.yaml}"

DATASET_DIR="${EXP_CAMPO_DATASET_DIR:-$ROOT/datasets/camponotus_rfdetr_coco}"
OUT_DIR="${EXP_CAMPO_OUT_DIR:-$ROOT/experiments/rfdetr/camponotus_rfdetr}"
GT_VAL="${EXP_CAMPO_GT_VAL:-$DATASET_DIR/valid/_annotations.coco.json}"
IMG_VAL="${EXP_CAMPO_IMG_VAL:-$DATASET_DIR/valid}"
PRED_VAL="${EXP_CAMPO_PRED_VAL:-$OUT_DIR/predictions_val.json}"
BENCH_VAL="${EXP_CAMPO_BENCH_VAL:-$OUT_DIR/inference_benchmark_val.json}"
GT_TEST="${EXP_CAMPO_GT_TEST:-$ROOT/datasets/camponotus_coco/annotations/instances_test.json}"
IMG_TEST="${EXP_CAMPO_IMG_TEST:-$ROOT/datasets/camponotus_yolo/images/test}"
PRED_TEST="${EXP_CAMPO_PRED_TEST:-$OUT_DIR/predictions_test.json}"
BENCH_TEST="${EXP_CAMPO_BENCH_TEST:-$OUT_DIR/inference_benchmark_test.json}"
MET_VAL="${EXP_CAMPO_MET_VAL:-$ROOT/experiments/results/camponotus_rfdetr_val_metrics.json}"
MET_TEST="${EXP_CAMPO_MET_TEST:-$ROOT/experiments/results/camponotus_rfdetr_test_metrics.json}"
CMP_VAL="${EXP_CAMPO_CMP_VAL:-$ROOT/experiments/results/camponotus_rfdetr_val_vs_yolo.json}"
CMP_TEST="${EXP_CAMPO_CMP_TEST:-$ROOT/experiments/results/camponotus_rfdetr_test_vs_yolo.json}"
YOLO_MET_VAL="${EXP_CAMPO_YOLO_MET_VAL:-$ROOT/experiments/results/camponotus_yolo26n_val_metrics.json}"
YOLO_MET_TEST="${EXP_CAMPO_YOLO_MET_TEST:-$ROOT/experiments/results/camponotus_yolo26n_test_metrics.json}"
EXP_ID_VAL="${EXP_CAMPO_EXP_ID_VAL:-EXP-CAMPO-RFDETR-VAL}"
EXP_ID_TEST="${EXP_CAMPO_EXP_ID_TEST:-EXP-CAMPO-RFDETR-TEST}"

MODEL_CLASS="$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1],encoding='utf-8'))['model_class'])" "$CFG" 2>/dev/null || echo RFDETRSmall)"
CONF_THR="$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1],encoding='utf-8'))['conf_threshold'])" "$CFG" 2>/dev/null || echo 0.25)"

WEIGHTS="$OUT_DIR/weights/best.pth"
TRAIN_CFG_OUT="$OUT_DIR/config.yaml"
MANIFEST="$DATASET_DIR/camponotus_rfdetr_manifest.json"

DEVICE="${RFDETR_DEVICE:-${CAMPO_DEVICE:-${SMOKE_DEVICE:-auto}}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo cpu)"
fi

echo "== EXP-CAMPO-RFDETR config: $CFG =="
echo "== device: $DEVICE =="

if [[ ! -f "$DATASET_DIR/valid/_annotations.coco.json" ]]; then
  echo "== Prepare camponotus_rfdetr_coco =="
  python3 "$ROOT/scripts/datasets/prepare_camponotus_coco_rfdetr.py" \
    --config "$ROOT/configs/datasets/camponotus_coco_rfdetr.yaml"
fi

if [[ "${EXP_CAMPO_SKIP_TRAIN:-0}" != "1" ]]; then
  echo "== Train RF-DETR → $OUT_DIR =="
  python3 "$ROOT/scripts/train/train_rfdetr_ants.py" --config "$CFG"
else
  echo "== Skip train (EXP_CAMPO_SKIP_TRAIN=1) =="
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing weights: $WEIGHTS (train or copy checkpoint here)" >&2
  exit 1
fi

MAX_ARGS=()
if [[ -n "${EXP_CAMPO_MAX_IMAGES:-}" ]]; then
  MAX_ARGS+=(--max-images "$EXP_CAMPO_MAX_IMAGES")
fi

_run_split() {
  local split_name="$1"
  local gt="$2"
  local imgdir="$3"
  local pred="$4"
  local bench="$5"
  local met="$6"
  local exp_id="$7"

  for f in "$gt" "$imgdir"; do
    if [[ ! -e "$f" ]]; then
      echo "Missing ($split_name): $f" >&2
      exit 1
    fi
  done

  echo "== Infer $split_name → $pred =="
  python3 "$ROOT/scripts/inference/infer_rfdetr.py" \
    --weights "$WEIGHTS" \
    --source "$imgdir" \
    --coco-gt "$gt" \
    --out "$pred" \
    --model-class "$MODEL_CLASS" \
    --conf "$CONF_THR" \
    --class-id-mode multiclass \
    --device "$DEVICE" \
    "${MAX_ARGS[@]}"

  echo "== Bench $split_name → $bench =="
  python3 "$ROOT/scripts/evaluation/bench_rfdetr.py" \
    --weights "$WEIGHTS" \
    --source "$imgdir" \
    --coco-gt "$gt" \
    --model-class "$MODEL_CLASS" \
    --conf "$CONF_THR" \
    --device "$DEVICE" \
    --out "$bench" \
    --config "$CFG" \
    "${MAX_ARGS[@]}"

  mkdir -p "$ROOT/experiments/results"
  echo "== Evaluate $split_name → $met =="
  EVAL_EXTRA=()
  if [[ -f "$MANIFEST" ]]; then
    EVAL_EXTRA+=(--prepare-manifest "$MANIFEST")
  fi
  python3 "$ROOT/scripts/evaluation/evaluate.py" \
    --gt "$gt" \
    --pred "$pred" \
    --weights "$WEIGHTS" \
    --images-dir "$imgdir" \
    --out "$met" \
    --experiment-id "$exp_id" \
    --train-config "$TRAIN_CFG_OUT" \
    "${EVAL_EXTRA[@]}" \
    --device "$DEVICE" \
    --inference-benchmark-json "$bench"
}

_run_split "val" "$GT_VAL" "$IMG_VAL" "$PRED_VAL" "$BENCH_VAL" "$MET_VAL" "$EXP_ID_VAL"
_run_split "test" "$GT_TEST" "$IMG_TEST" "$PRED_TEST" "$BENCH_TEST" "$MET_TEST" "$EXP_ID_TEST"

echo "== Compare val vs YOLO → $CMP_VAL =="
if [[ -f "$YOLO_MET_VAL" ]]; then
  python3 "$ROOT/scripts/evaluation/compare_camponotus_rfdetr_vs_yolo.py" \
    --baseline "$YOLO_MET_VAL" \
    --compare "$MET_VAL" \
    --out "$CMP_VAL"
else
  echo "(Skip val compare: baseline not found at $YOLO_MET_VAL)"
fi

echo "== Compare test vs YOLO → $CMP_TEST =="
if [[ -f "$YOLO_MET_TEST" ]]; then
  python3 "$ROOT/scripts/evaluation/compare_camponotus_rfdetr_vs_yolo.py" \
    --baseline "$YOLO_MET_TEST" \
    --compare "$MET_TEST" \
    --out "$CMP_TEST"
else
  echo "(Skip test compare: baseline not found at $YOLO_MET_TEST)"
fi

echo "EXP-CAMPO-RFDETR finished."
