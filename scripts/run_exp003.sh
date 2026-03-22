#!/usr/bin/env bash
# EXP-003: SAHI sliced inference on val (no training). Compare vs vanilla test_run + exp002b_imgsz896.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${EXP003_DEVICE:-${SMOKE_DEVICE:-auto}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print(0 if torch.cuda.is_available() else \"cpu\")" 2>/dev/null || echo cpu)"
fi

SAHI_CFG="${EXP003_SAHI_CONFIG:-$ROOT/configs/exp003_sahi.yaml}"
GT_VAL="$ROOT/datasets/processed/test_run/annotations/instances_val.json"
IMG_VAL="$ROOT/datasets/processed/test_run/images/val"
MANIFEST="$ROOT/datasets/processed/test_run/prepare_manifest.json"

W_BASE="$ROOT/experiments/yolo/test_run/weights/best.pt"
[[ -f "$W_BASE" ]] || W_BASE="$ROOT/experiments/yolo/test_run/weights/last.pt"
W_896="$ROOT/experiments/yolo/exp002b_imgsz896/weights/best.pt"
[[ -f "$W_896" ]] || W_896="$ROOT/experiments/yolo/exp002b_imgsz896/weights/last.pt"
TC_BASE="$ROOT/experiments/yolo/test_run/config.yaml"
TC_896="$ROOT/experiments/yolo/exp002b_imgsz896/config.yaml"

MET_BASE="$ROOT/experiments/results/test_run_metrics.json"
MET_896="$ROOT/experiments/results/exp002b_imgsz896_metrics.json"

OUT_Y_BASE="$ROOT/experiments/yolo/test_run_exp003_sahi_base"
OUT_Y_896="$ROOT/experiments/yolo/test_run_exp003_sahi_896"
PRED_BASE="$OUT_Y_BASE/predictions_val.json"
PRED_896="$OUT_Y_896/predictions_val.json"
M_SAHI_BASE="$ROOT/experiments/results/test_run_exp003_sahi_base_metrics.json"
M_SAHI_896="$ROOT/experiments/results/test_run_exp003_sahi_896_metrics.json"

echo "== EXP-003 device: $DEVICE (EXP003_DEVICE or SMOKE_DEVICE) =="

for f in "$W_BASE" "$W_896" "$GT_VAL" "$SAHI_CFG" "$MET_BASE" "$MET_896"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f" >&2
    if [[ "$f" == "$MET_896" ]] || [[ "$f" == "$W_896" ]]; then
      echo "Run EXP-002b first (./scripts/run_exp002b.sh) so exp002b_imgsz896 weights and metrics exist." >&2
    fi
    exit 1
  fi
done

mkdir -p "$OUT_Y_BASE" "$OUT_Y_896" "$ROOT/experiments/results"

echo "== SAHI infer (test_run weights) → $PRED_BASE =="
python3 "$ROOT/scripts/inference/infer_sahi_yolo.py" \
  --weights "$W_BASE" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$PRED_BASE" \
  --sahi-config "$SAHI_CFG" \
  --device "$DEVICE"

echo "== SAHI infer (exp002b_imgsz896 weights) → $PRED_896 =="
python3 "$ROOT/scripts/inference/infer_sahi_yolo.py" \
  --weights "$W_896" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$PRED_896" \
  --sahi-config "$SAHI_CFG" \
  --device "$DEVICE"

echo "== Evaluate SAHI base → $M_SAHI_BASE =="
python3 "$ROOT/scripts/evaluation/evaluate.py" \
  --gt "$GT_VAL" \
  --pred "$PRED_BASE" \
  --out "$M_SAHI_BASE" \
  --weights "$W_BASE" \
  --images-dir "$IMG_VAL" \
  --device "$DEVICE" \
  --experiment-id "EXP-003-sahi-base" \
  --prepare-manifest "$MANIFEST" \
  --train-config "$TC_BASE" \
  --sahi-config "$SAHI_CFG"

echo "== Evaluate SAHI 896-weights → $M_SAHI_896 =="
python3 "$ROOT/scripts/evaluation/evaluate.py" \
  --gt "$GT_VAL" \
  --pred "$PRED_896" \
  --out "$M_SAHI_896" \
  --weights "$W_896" \
  --images-dir "$IMG_VAL" \
  --device "$DEVICE" \
  --experiment-id "EXP-003-sahi-896" \
  --prepare-manifest "$MANIFEST" \
  --train-config "$TC_896" \
  --sahi-config "$SAHI_CFG"

NOTE_BASE="SAHI sliced inference vs vanilla EXP-000 (test_run); same val GT; training unchanged. Vanilla benchmark used plain YOLO predict; SAHI row uses sliced FPS from evaluate.py --sahi-config."
NOTE896="SAHI sliced inference with weights trained at imgsz=896 (EXP-002b) vs vanilla 896 predict; same val GT."

echo "== Compare SAHI base vs test_run_metrics =="
python3 "$ROOT/scripts/evaluation/compare_metrics.py" \
  --baseline "$MET_BASE" \
  --compare "$M_SAHI_BASE" \
  --out "$ROOT/experiments/results/exp003_sahi_vs_baseline.json" \
  --summary-label "EXP-003 SAHI (test_run weights) vs EXP-000 vanilla" \
  --evaluation-note "$NOTE_BASE"

echo "== Compare SAHI 896-weights vs exp002b_imgsz896 metrics =="
python3 "$ROOT/scripts/evaluation/compare_metrics.py" \
  --baseline "$MET_896" \
  --compare "$M_SAHI_896" \
  --out "$ROOT/experiments/results/exp003_sahi_vs_exp002b_896.json" \
  --summary-label "EXP-003 SAHI (896-trained weights) vs vanilla 896" \
  --evaluation-note "$NOTE896"

echo "== Visualizations =="
python3 "$ROOT/scripts/visualization/viz_coco_overlays.py" \
  --pred "$PRED_BASE" \
  --gt "$GT_VAL" \
  --images-dir "$IMG_VAL" \
  --out-dir "$ROOT/experiments/visualizations/test_run_exp003_sahi_base"

python3 "$ROOT/scripts/visualization/viz_coco_overlays.py" \
  --pred "$PRED_896" \
  --gt "$GT_VAL" \
  --images-dir "$IMG_VAL" \
  --out-dir "$ROOT/experiments/visualizations/test_run_exp003_sahi_896"

echo "== Summary markdown =="
python3 "$ROOT/scripts/evaluation/write_exp003_sahi_summary.py" \
  --compare-baseline "$ROOT/experiments/results/exp003_sahi_vs_baseline.json" \
  --compare-896 "$ROOT/experiments/results/exp003_sahi_vs_exp002b_896.json" \
  --out "$ROOT/experiments/results/exp003_sahi_summary.md"

echo "EXP-003 finished."
