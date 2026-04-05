#!/usr/bin/env bash
# CVAT bundle → align → track_id-majority split → YOLO+COCO → RF-DETR layout → optional YOLO then RF-DETR train.
# Usage:
#   export FULL_ROOT="/path/to/cvat bundle"   # must contain images/ and annotations/
#   ./scripts/run_camponotus_cvat_bundle_prepare_train.sh           # prepare + verify only
#   ./scripts/run_camponotus_cvat_bundle_prepare_train.sh --train   # also train YOLO then RF-DETR
#
# Override paths with env: CVAT_COCO, REPO, ALIGNED, SPLITS_TM, YOLO_OUT, COCO_OUT, RFDETR_OUT
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FULL_ROOT="${FULL_ROOT:-/media/dmytro/data/datasets/camponotus fellah trophallaxis FULL dataset}"
CVAT_COCO="${CVAT_COCO:-$FULL_ROOT/annotations/instances_default.json}"
REPO="${REPO:-$ROOT}"

ALIGNED="${ALIGNED:-$REPO/datasets/camponotus_processed/camponotus_full_cvat_aligned.json}"
SPLITS_TM="${SPLITS_TM:-$REPO/datasets/camponotus_processed/splits_trackid_majority_full_export_unique.json}"
YOLO_OUT="${YOLO_OUT:-$REPO/datasets/camponotus_yolo/camponotus_full_export_unique_trackidmajor}"
COCO_OUT="${COCO_OUT:-$REPO/datasets/camponotus_coco/camponotus_full_export_unique_trackidmajor}"
RFDETR_OUT="${RFDETR_OUT:-$REPO/datasets/camponotus_rfdetr_coco_trackidmajor}"
ANALYSIS_OUT="${ANALYSIS_OUT:-$REPO/datasets/camponotus_processed/analysis.json}"

TRAIN=0
for a in "$@"; do
  if [[ "$a" == "--train" ]]; then
    TRAIN=1
  fi
done

echo "== Camponotus CVAT bundle pipeline =="
echo "FULL_ROOT=$FULL_ROOT"
echo "CVAT_COCO=$CVAT_COCO"

if [[ ! -f "$CVAT_COCO" ]]; then
  echo "Missing CVAT COCO: $CVAT_COCO" >&2
  exit 1
fi
if [[ ! -d "$FULL_ROOT" ]]; then
  echo "Missing FULL_ROOT: $FULL_ROOT" >&2
  exit 1
fi

mkdir -p "$(dirname "$ALIGNED")"

echo "== 1) Align COCO file_name → paths under FULL_ROOT =="
python3 "$REPO/scripts/datasets/align_coco_filenames_to_camponotus_raw.py" \
  --coco "$CVAT_COCO" \
  --raw-root "$FULL_ROOT" \
  --out "$ALIGNED"

echo "== 2) Verify aligned COCO vs disk =="
python3 "$REPO/scripts/datasets/verify_camponotus_cvat_bundle.py" \
  --coco "$ALIGNED" \
  --raw-root "$FULL_ROOT"

echo "== 3) track_id–majority split manifest =="
python3 "$REPO/scripts/datasets/split_camponotus_dataset_by_track_id_majority.py" \
  --coco-json "$ALIGNED" \
  --out "$SPLITS_TM" \
  --seed 42

echo "== 4) YOLO + COCO export =="
python3 "$REPO/scripts/datasets/prepare_camponotus_detection_dataset.py" \
  --coco-annotations "$ALIGNED" \
  --split-source manifest \
  --splits "$SPLITS_TM" \
  --raw-root "$FULL_ROOT" \
  --out-yolo "$YOLO_OUT" \
  --out-coco "$COCO_OUT" \
  --analysis-out "$ANALYSIS_OUT" \
  --copy-mode symlink

echo "== 5) Validate YOLO export (labels, nc) =="
python3 "$REPO/scripts/datasets/validate_camponotus_dataset.py" \
  --yolo-root "$YOLO_OUT" \
  --coco-root "$COCO_OUT/annotations" \
  --analysis-json "$ANALYSIS_OUT"

python3 "$REPO/scripts/datasets/verify_camponotus_cvat_bundle.py" \
  --prepared-yolo "$YOLO_OUT"

echo "== 6) RF-DETR Roboflow-style COCO =="
python3 "$REPO/scripts/datasets/prepare_camponotus_coco_rfdetr.py" \
  --camponotus-yolo-root "$YOLO_OUT" \
  --camponotus-coco-annotations-root "$COCO_OUT/annotations" \
  --out-root "$RFDETR_OUT"

python3 "$REPO/scripts/datasets/verify_camponotus_cvat_bundle.py" \
  --prepared-rfdetr "$RFDETR_OUT"

echo "== Prepare + verify complete. =="
echo "  Aligned COCO: $ALIGNED"
echo "  YOLO:         $YOLO_OUT"
echo "  COCO splits:  $COCO_OUT"
echo "  RF-DETR:      $RFDETR_OUT"

if [[ "$TRAIN" != 1 ]]; then
  echo ""
  echo "To train YOLO then RF-DETR, re-run with --train or run manually:"
  echo "  python3 scripts/train/train_yolo.py --config-name=train/yolo_camponotus_trackidmajor_n896"
  echo "  python3 scripts/train/train_rfdetr_ants.py --config configs/camponotus_rfdetr_trackidmajor_896_long_train.yaml"
  exit 0
fi

echo "== 7) Train YOLO26n @896 =="
python3 "$REPO/scripts/train/train_yolo.py" --config-name=train/yolo_camponotus_trackidmajor_n896

echo "== 8) Train RF-DETR Small @896 (long schedule) =="
python3 "$REPO/scripts/train/train_rfdetr_ants.py" \
  --config "$REPO/configs/camponotus_rfdetr_trackidmajor_896_long_train.yaml"

echo "== Done. Weights (typical): =="
echo "  YOLO:    experiments/yolo/camponotus_trackidmajor_full_896/weights/best.pt"
echo "  RF-DETR: experiments/rfdetr/camponotus_rfdetr_trackidmajor_896_ep60_es/weights/best.pth"
