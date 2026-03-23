#!/usr/bin/env bash
# EXP-A004 (ants): ANTS v1 two-stage inference vs vanilla 768 + optional SAHI compare. No training.
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
MET_768="$ROOT/experiments/results/ants_expA002b_imgsz768_metrics.json"
PRED_BASELINE="$YOLO_768/predictions_val.json"

OUT_DIR="$ROOT/experiments/yolo/ants_expA004"
PRED_ANTS="$OUT_DIR/predictions_val.json"
BENCH_JSON="$OUT_DIR/inference_benchmark.json"
MET_ANTS="${EXP_A004_METRICS_OUT:-$ROOT/experiments/results/ants_expA004_ants_metrics.json}"
CMP_OUT="${EXP_A004_COMPARE_OUT:-$ROOT/experiments/results/ants_expA004_vs_baseline.json}"
EXP_ID="${EXP_A004_EXPERIMENT_ID:-EXP-A004-ANTS-v1}"
VIZ_ROOT="$ROOT/experiments/visualizations/ants_expA004"
SUMMARY_OUT="${EXP_A004_SUMMARY_OUT:-$ROOT/experiments/results/ants_expA004_summary.md}"

MAX_IMG_ARGS=()
if [[ -n "${EXP_A004_MAX_IMAGES:-}" ]]; then
  MAX_IMG_ARGS+=(--max-images "$EXP_A004_MAX_IMAGES")
fi

echo "== EXP-A004 device: $DEVICE (ANTS_DEVICE or SMOKE_DEVICE) =="
echo "== ANTS config: $ANTS_CFG (override with EXP_A004_ANT_CONFIG) =="

for f in "$GT_VAL" "$IMG_VAL" "$MANIFEST" "$WEIGHTS" "$TC_768" "$MET_768" "$ANTS_CFG"; do
  if [[ ! -e "$f" ]]; then
    echo "Missing required path: $f" >&2
    if [[ "$f" == "$MET_768" ]] || [[ "$f" == "$WEIGHTS" ]]; then
      echo "Run EXP-A002b first so ants_expA002b_imgsz768 weights and metrics exist." >&2
    fi
    exit 1
  fi
done

mkdir -p "$OUT_DIR" "$ROOT/experiments/results" "$ROOT/experiments/visualizations"

cp -f "$ANTS_CFG" "$OUT_DIR/config.yaml"

python3 <<PY
import json
import platform
import sys
from pathlib import Path

import torch

out = Path("$OUT_DIR") / "system_info.json"
info = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    info["cuda_device_count"] = torch.cuda.device_count()
    info["cuda_device_0"] = torch.cuda.get_device_name(0)
out.write_text(json.dumps(info, indent=2), encoding="utf-8")
print("Wrote", out)
PY

echo "== ANTS infer → $PRED_ANTS =="
python3 "$ROOT/scripts/inference/infer_ants_v1.py" \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$PRED_ANTS" \
  --config "$ANTS_CFG" \
  --device "$DEVICE" \
  --rois-out "$OUT_DIR/rois.json" \
  --stage1-out "$OUT_DIR/predictions_stage1_val.json" \
  "${MAX_IMG_ARGS[@]}"

echo "== Bench full ANTS pipeline → $BENCH_JSON =="
python3 "$ROOT/scripts/evaluation/bench_ants_v1.py" \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --config "$ANTS_CFG" \
  --device "$DEVICE" \
  --out "$BENCH_JSON" \
  "${MAX_IMG_ARGS[@]}"

echo "== Evaluate ANTS → $MET_ANTS =="
python3 "$ROOT/scripts/evaluation/evaluate.py" \
  --gt "$GT_VAL" \
  --pred "$PRED_ANTS" \
  --out "$MET_ANTS" \
  --weights "$WEIGHTS" \
  --images-dir "$IMG_VAL" \
  --device "$DEVICE" \
  --experiment-id "$EXP_ID" \
  --prepare-manifest "$MANIFEST" \
  --train-config "$TC_768" \
  --inference-benchmark-json "$BENCH_JSON"

echo "== Compare vs 768 (+ SAHI if present) → $CMP_OUT =="
python3 "$ROOT/scripts/evaluation/compare_ants_expA004.py" \
  --metrics-768 "$MET_768" \
  --metrics-sahi "$ROOT/experiments/results/ants_expA003_sahi_metrics.json" \
  --metrics-ants "$MET_ANTS" \
  --out "$CMP_OUT"

VIZ_ARGS=()
if [[ "${ANTS_VIZ_MAX_IMAGES:-}" == "all" ]]; then
  :
elif [[ -n "${ANTS_VIZ_MAX_IMAGES:-}" ]]; then
  VIZ_ARGS+=(--max-images "$ANTS_VIZ_MAX_IMAGES")
else
  VIZ_ARGS+=(--max-images 250)
fi

mkdir -p "$VIZ_ROOT/baseline_comparisons" "$VIZ_ROOT/ants_comparisons" "$VIZ_ROOT/rois_debug"

if [[ -f "$PRED_BASELINE" ]]; then
  echo "== Viz baseline 768 comparisons → $VIZ_ROOT/baseline_comparisons =="
  python3 "$ROOT/scripts/visualization/viz_ants_expA004_comparisons.py" \
    --pred "$PRED_BASELINE" \
    --gt "$GT_VAL" \
    --images-dir "$IMG_VAL" \
    --out-dir "$VIZ_ROOT/baseline_comparisons" \
    "${VIZ_ARGS[@]}"
else
  echo "== Skip baseline viz: missing $PRED_BASELINE (run infer on 768 run if needed) =="
fi

echo "== Viz ANTS comparisons → $VIZ_ROOT/ants_comparisons =="
python3 "$ROOT/scripts/visualization/viz_ants_expA004_comparisons.py" \
  --pred "$PRED_ANTS" \
  --gt "$GT_VAL" \
  --images-dir "$IMG_VAL" \
  --out-dir "$VIZ_ROOT/ants_comparisons" \
  "${VIZ_ARGS[@]}"

ROI_VIZ_ARGS=()
if [[ -n "${ANTS_ROI_VIZ_MAX_IMAGES:-}" ]]; then
  ROI_VIZ_ARGS+=(--max-images "$ANTS_ROI_VIZ_MAX_IMAGES")
else
  ROI_VIZ_ARGS+=(--max-images 50)
fi

echo "== Viz ROIs → $VIZ_ROOT/rois_debug =="
python3 "$ROOT/scripts/visualization/viz_ants_rois.py" \
  --rois-json "$OUT_DIR/rois.json" \
  --images-dir "$IMG_VAL" \
  --out-dir "$VIZ_ROOT/rois_debug" \
  "${ROI_VIZ_ARGS[@]}"

echo "== Summary → $SUMMARY_OUT =="
python3 "$ROOT/scripts/evaluation/write_ants_expA004_summary.py" \
  --compare "$CMP_OUT" \
  --rois-json "$OUT_DIR/rois.json" \
  --out "$SUMMARY_OUT"

echo "EXP-A004 (ANTS v1) finished."
