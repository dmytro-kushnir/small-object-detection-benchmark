#!/usr/bin/env bash
# EXP-A002b: resolution sweep (640, 768, 896, 1024) on datasets/ants_yolo — same val GT as EXP-A000 full.
# Prerequisite: ./scripts/run_ants_prepare.sh (ANTS_DATASET_ROOT). Optional: completed ants_expA000_full for 640 reuse.
# If VRAM OOM: EXP_A002B_BATCH=2 ./scripts/run_ants_expA002b.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${ANTS_DEVICE:-${SMOKE_DEVICE:-auto}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print(0 if torch.cuda.is_available() else \"cpu\")" 2>/dev/null || echo cpu)"
fi
BATCH="${EXP_A002B_BATCH:-4}"
echo "== EXP-A002b device: $DEVICE (ANTS_DEVICE or SMOKE_DEVICE); batch: $BATCH (EXP_A002B_BATCH) =="

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

_full_train_imgsz() {
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
" "$ROOT/experiments/yolo/ants_expA000_full/config.yaml"
}

BASELINE_METRICS="$ROOT/experiments/results/ants_expA000_full_metrics.json"
BASELINE_CFG="$ROOT/experiments/yolo/ants_expA000_full/config.yaml"
FI="$(_full_train_imgsz || true)"
REUSE_640=0
if [[ "$FI" == "640" ]] && [[ -f "$BASELINE_METRICS" ]] && [[ -f "$BASELINE_CFG" ]]; then
  REUSE_640=1
  echo "== Will reuse EXP-A000 full artifacts for imgsz 640 (copy metrics; preds from ants_expA000_full) =="
fi

for SZ in 640 768 896 1024; do
  NAME="ants_expA002b_imgsz${SZ}"
  YDIR="$ROOT/experiments/yolo/$NAME"
  MET="$ROOT/experiments/results/${NAME}_metrics.json"
  EID="EXP-A002b-imgsz${SZ}"

  echo ""
  echo "########## EXP-A002b imgsz=${SZ} ##########"

  if [[ "$SZ" == "640" ]] && [[ "$REUSE_640" -eq 1 ]]; then
    python3 -c "
import json
from pathlib import Path
src = Path('${ROOT}') / 'experiments/results/ants_expA000_full_metrics.json'
dst = Path('${ROOT}') / 'experiments/results/ants_expA002b_imgsz640_metrics.json'
data = json.loads(src.read_text(encoding='utf-8'))
data['experiment_id'] = 'EXP-A002b-imgsz640'
data['reuse_note'] = 'metrics copied from ants_expA000_full (same 640 train/infer)'
dst.write_text(json.dumps(data, indent=2), encoding='utf-8')
print('Wrote', dst)
"
    echo "Skipped train/infer for 640 (reused baseline metrics)."
    continue
  fi

  echo "== Train → $YDIR =="
  python3 "$ROOT/scripts/train/train_yolo.py" --config-name=train/yolo_ants_expA002b \
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
  python3 "$ROOT/scripts/inference/infer_yolo.py" \
    --weights "$WEIGHTS" \
    --source "$IMG_VAL" \
    --coco-gt "$GT_VAL" \
    --out "$PRED" \
    --device "$DEVICE" \
    --imgsz "$SZ"

  echo "== Evaluation → $MET =="
  python3 "$ROOT/scripts/evaluation/evaluate.py" \
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

# Per-resolution viz (FP/FN in comparisons/)
VIZ_ARGS=()
if [[ "${ANTS_VIZ_MAX_IMAGES:-}" == "all" ]]; then
  :
elif [[ -n "${ANTS_VIZ_MAX_IMAGES:-}" ]]; then
  VIZ_ARGS+=(--max-images "$ANTS_VIZ_MAX_IMAGES")
else
  VIZ_ARGS+=(--max-images 250)
fi

for SZ in 640 768 896 1024; do
  NAME="ants_expA002b_imgsz${SZ}"
  if [[ "$SZ" == "640" ]] && [[ "$REUSE_640" -eq 1 ]]; then
    PRED="$ROOT/experiments/yolo/ants_expA000_full/predictions_val.json"
  else
    PRED="$ROOT/experiments/yolo/$NAME/predictions_val.json"
  fi
  if [[ ! -f "$PRED" ]]; then
    echo "Skip viz imgsz=$SZ: missing $PRED" >&2
    continue
  fi
  VOUT="$ROOT/experiments/visualizations/ants_expA002b/imgsz${SZ}"
  echo "== Viz imgsz=$SZ → $VOUT =="
  python3 "$ROOT/scripts/visualization/viz_coco_overlays.py" \
    --pred "$PRED" \
    --gt "$GT_VAL" \
    --images-dir "$IMG_VAL" \
    --out-dir "$VOUT" \
    "${VIZ_ARGS[@]}"
done

echo ""
echo "== Summarize sweep → experiments/results/ants_expA002b_resolution_sweep.json =="
python3 "$ROOT/scripts/evaluation/summarize_ants_resolution_sweep.py" --cwd "$ROOT"

REL_OUT="$ROOT/experiments/results/ants_expA002b_relative_metrics.json"
echo "== Relative metrics aggregate → $REL_OUT =="
PRED_ARGS=()
for SZ in 640 768 896 1024; do
  NAME="ants_expA002b_imgsz${SZ}"
  if [[ "$SZ" == "640" ]] && [[ "$REUSE_640" -eq 1 ]]; then
    PP="$ROOT/experiments/yolo/ants_expA000_full/predictions_val.json"
  else
    PP="$ROOT/experiments/yolo/$NAME/predictions_val.json"
  fi
  if [[ -f "$PP" ]]; then
    PRED_ARGS+=(--pred "${SZ}=${PP}")
  fi
done
if [[ ${#PRED_ARGS[@]} -gt 0 ]]; then
  python3 "$ROOT/scripts/evaluation/ants_relative_sweep_aggregate.py" \
    --coco-gt "$GT_VAL" \
    --out "$REL_OUT" \
    "${PRED_ARGS[@]}"
else
  echo "No prediction files found; skip relative aggregate." >&2
fi

echo "== Summary markdown → experiments/results/ants_expA002b_summary.md =="
python3 "$ROOT/scripts/evaluation/write_ants_expA002b_summary.py" \
  --sweep "$ROOT/experiments/results/ants_expA002b_resolution_sweep.json" \
  --baseline-metrics "$ROOT/experiments/results/ants_expA000_full_metrics.json" \
  --relative "$REL_OUT" \
  --out "$ROOT/experiments/results/ants_expA002b_summary.md"

echo "EXP-A002b finished."
