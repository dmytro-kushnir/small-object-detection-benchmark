#!/usr/bin/env bash
# Camponotus dataset workflow (data prep only; no training).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CFG="${CAMPO_CFG:-$ROOT/configs/datasets/camponotus_workflow.yaml}"
if [[ ! -f "$CFG" ]]; then
  echo "Missing config: $CFG" >&2
  exit 1
fi

read_cfg() {
  python3 - "$CFG" "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8')) or {}
cur = cfg
for k in sys.argv[2].split("."):
    if not isinstance(cur, dict):
        cur = None
        break
    cur = cur.get(k)
print("" if cur is None else cur)
PY
}

RAW_ROOT="$ROOT/$(read_cfg paths.raw_root)"
IN_SITU_ROOT="$ROOT/$(read_cfg paths.in_situ_root)"
EXT_DIR="$ROOT/$(read_cfg paths.external_images_dir)"
SPLITS="$ROOT/$(read_cfg paths.split_manifest)"
CVAT_COCO="$ROOT/$(read_cfg paths.cvat_coco_annotations)"
YOLO_ROOT="$ROOT/$(read_cfg paths.yolo_root)"
COCO_ROOT="$ROOT/$(read_cfg paths.coco_root)"
ANALYSIS="$ROOT/$(read_cfg paths.analysis_json)"
TR="$(read_cfg split.train_ratio)"
VR="$(read_cfg split.val_ratio)"
TER="$(read_cfg split.test_ratio)"
SEED="$(read_cfg split.seed)"

echo "== Split sequences =="
python3 "$ROOT/scripts/datasets/split_camponotus_dataset.py" \
  --in-situ-root "$IN_SITU_ROOT" \
  --external-images-dir "$EXT_DIR" \
  --train-ratio "$TR" \
  --val-ratio "$VR" \
  --test-ratio "$TER" \
  --seed "$SEED" \
  --out "$SPLITS"

if [[ ! -f "$CVAT_COCO" ]]; then
  echo "Missing corrected CVAT COCO annotations: $CVAT_COCO" >&2
  echo "Export from CVAT first (see docs/camponotus_cvat_workflow.md)." >&2
  exit 1
fi

echo "== Convert to YOLO + COCO =="
python3 "$ROOT/scripts/datasets/prepare_camponotus_detection_dataset.py" \
  --coco-annotations "$CVAT_COCO" \
  --splits "$SPLITS" \
  --raw-root "$RAW_ROOT" \
  --out-yolo "$YOLO_ROOT" \
  --out-coco "$COCO_ROOT" \
  --analysis-out "$ANALYSIS"

echo "== Analyze =="
python3 "$ROOT/scripts/datasets/analyze_camponotus_dataset.py" \
  --coco-root "$COCO_ROOT/annotations" \
  --split-manifest "$SPLITS" \
  --out-json "$ANALYSIS" \
  --plots-dir "$ROOT/datasets/camponotus_processed/plots"

echo "== Validate =="
python3 "$ROOT/scripts/datasets/validate_camponotus_dataset.py" \
  --yolo-root "$YOLO_ROOT" \
  --coco-root "$COCO_ROOT/annotations" \
  --analysis-json "$ANALYSIS"

echo "== Visualize samples =="
python3 "$ROOT/scripts/visualization/viz_camponotus_dataset_samples.py" \
  --dataset-yaml "$YOLO_ROOT/dataset.yaml" \
  --out-dir "$ROOT/experiments/visualizations/camponotus_dataset"

echo "Camponotus dataset workflow completed."
