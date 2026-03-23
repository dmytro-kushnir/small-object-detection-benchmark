#!/usr/bin/env bash
# EXP-A003 (ants): SAHI sliced val inference vs vanilla imgsz=768 (ants_expA002b_imgsz768). No training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${ANTS_DEVICE:-${SMOKE_DEVICE:-auto}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print(0 if torch.cuda.is_available() else \"cpu\")" 2>/dev/null || echo cpu)"
fi

SAHI_CFG="${EXP_A003_SAHI_CONFIG:-$ROOT/configs/expA003_ants_sahi.yaml}"
GT_VAL="$ROOT/datasets/ants_yolo/annotations/instances_val.json"
IMG_VAL="$ROOT/datasets/ants_yolo/images/val"
MANIFEST="$ROOT/datasets/ants_yolo/prepare_manifest.json"

YOLO_768="$ROOT/experiments/yolo/ants_expA002b_imgsz768"
WEIGHTS="$YOLO_768/weights/best.pt"
[[ -f "$WEIGHTS" ]] || WEIGHTS="$YOLO_768/weights/last.pt"
TC_768="$YOLO_768/config.yaml"
MET_768="$ROOT/experiments/results/ants_expA002b_imgsz768_metrics.json"

OUT_DIR="$ROOT/experiments/yolo/ants_expA003_sahi"
PRED_SAHI="$OUT_DIR/predictions_val.json"
MET_SAHI="$ROOT/experiments/results/ants_expA003_sahi_metrics.json"
CMP_OUT="$ROOT/experiments/results/ants_expA003_vs_768.json"
VIZ_OUT="$ROOT/experiments/visualizations/ants_expA003_sahi"
SUMMARY_OUT="$ROOT/experiments/results/ants_expA003_summary.md"

DATA_YAML="$ROOT/datasets/ants_yolo/dataset.yaml"

echo "== EXP-A003 device: $DEVICE (ANTS_DEVICE or SMOKE_DEVICE) =="
echo "== SAHI config: $SAHI_CFG (override with EXP_A003_SAHI_CONFIG) =="

for f in "$DATA_YAML" "$GT_VAL" "$IMG_VAL" "$MANIFEST" "$WEIGHTS" "$TC_768" "$MET_768" "$SAHI_CFG"; do
  if [[ ! -e "$f" ]]; then
    echo "Missing required path: $f" >&2
    if [[ "$f" == "$MET_768" ]] || [[ "$f" == "$WEIGHTS" ]]; then
      echo "Run EXP-A002b first so ants_expA002b_imgsz768 weights and metrics exist." >&2
    fi
    exit 1
  fi
done

mkdir -p "$OUT_DIR" "$ROOT/experiments/results" "$ROOT/experiments/visualizations"

NOTE="Same ants val GT and images as EXP-A002b. Baseline metrics use vanilla Ultralytics predict at imgsz=768; SAHI metrics use evaluate.py --sahi-config (sliced FPS/latency per full image). Interpret speed deltas cautiously (different code paths)."

echo "== SAHI infer → $PRED_SAHI =="
python3 "$ROOT/scripts/inference/infer_sahi_yolo.py" \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$PRED_SAHI" \
  --sahi-config "$SAHI_CFG" \
  --device "$DEVICE"

echo "== Evaluate SAHI → $MET_SAHI =="
python3 "$ROOT/scripts/evaluation/evaluate.py" \
  --gt "$GT_VAL" \
  --pred "$PRED_SAHI" \
  --out "$MET_SAHI" \
  --weights "$WEIGHTS" \
  --images-dir "$IMG_VAL" \
  --device "$DEVICE" \
  --experiment-id "EXP-A003-sahi" \
  --prepare-manifest "$MANIFEST" \
  --train-config "$TC_768" \
  --imgsz 768 \
  --sahi-config "$SAHI_CFG"

echo "== Compare SAHI vs vanilla 768 → $CMP_OUT =="
python3 "$ROOT/scripts/evaluation/compare_metrics.py" \
  --baseline "$MET_768" \
  --compare "$MET_SAHI" \
  --out "$CMP_OUT" \
  --summary-label "EXP-A003 SAHI vs EXP-A002b vanilla imgsz=768" \
  --evaluation-note "$NOTE"

VIZ_ARGS=()
if [[ "${ANTS_VIZ_MAX_IMAGES:-}" == "all" ]]; then
  :
elif [[ -n "${ANTS_VIZ_MAX_IMAGES:-}" ]]; then
  VIZ_ARGS+=(--max-images "$ANTS_VIZ_MAX_IMAGES")
else
  VIZ_ARGS+=(--max-images 250)
fi

echo "== Viz → $VIZ_OUT =="
python3 "$ROOT/scripts/visualization/viz_coco_overlays.py" \
  --pred "$PRED_SAHI" \
  --gt "$GT_VAL" \
  --images-dir "$IMG_VAL" \
  --out-dir "$VIZ_OUT" \
  "${VIZ_ARGS[@]}"

echo "== Summary → $SUMMARY_OUT =="
python3 "$ROOT/scripts/evaluation/write_ants_expA003_summary.py" \
  --compare "$CMP_OUT" \
  --baseline-metrics "$MET_768" \
  --sahi-metrics "$MET_SAHI" \
  --out "$SUMMARY_OUT"

echo "EXP-A003 (ants) finished."
