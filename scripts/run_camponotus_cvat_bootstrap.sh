#!/usr/bin/env bash
# One-command CVAT bootstrap for Camponotus:
# 1) extract frames with FPS policy
# 2) generate tracked prelabels on extracted frames
# 3) convert to CVAT single-label (ant + state attribute) JSON
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VIDEOS_ROOT="${VIDEOS_ROOT:-}"
if [[ -z "$VIDEOS_ROOT" ]]; then
  echo "Set VIDEOS_ROOT to your videos folder path." >&2
  echo "Example: VIDEOS_ROOT=/media/dmytro/data/datasets/ants_videos $0" >&2
  exit 1
fi

YOLO_WEIGHTS="${YOLO_WEIGHTS:-$ROOT/experiments/yolo/camponotus_yolo26n_v2/weights/best.pt}"
OUT_RAW_ROOT="${OUT_RAW_ROOT:-$ROOT/datasets/camponotus_raw/in_situ}"
OUT_PRELABELS="${OUT_PRELABELS:-$ROOT/datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_botsort_reid_soft.json}"
OUT_CVAT="${OUT_CVAT:-$ROOT/datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_botsort_reid_soft_cvat_ant_only.json}"

FPS_DEFAULT="${FPS_DEFAULT:-5}"
FPS_TROPH="${FPS_TROPH:-8}"
IMAGE_EXT="${IMAGE_EXT:-.jpg}"

CONF="${CONF:-0.45}"
TRACK_THRESH="${TRACK_THRESH:-0.25}"
MATCH_THRESH="${MATCH_THRESH:-0.8}"
TRACK_BUFFER="${TRACK_BUFFER:-30}"
MIN_TRACK_LEN="${MIN_TRACK_LEN:-2}"
STATE_IOU_THRESH="${STATE_IOU_THRESH:-0.70}"
STATE_SCORE_GAP_MAX="${STATE_SCORE_GAP_MAX:-0.12}"

echo "== 1/3 Extract frames with FPS policy =="
python3 "$ROOT/scripts/datasets/extract_camponotus_frames.py" \
  --videos-root "$VIDEOS_ROOT" \
  --out-root "$OUT_RAW_ROOT" \
  --fps "$FPS_DEFAULT" \
  --fps-trophallaxis "$FPS_TROPH" \
  --seq-prefix seq_ \
  --seq-naming video \
  --image-ext "$IMAGE_EXT"

echo "== 2/3 Generate tracked prelabels on extracted frames =="
python3 "$ROOT/scripts/datasets/bootstrap_camponotus_autolabel.py" \
  --images-root "$OUT_RAW_ROOT" \
  --backend yolo \
  --yolo-weights "$YOLO_WEIGHTS" \
  --conf "$CONF" \
  --with-tracking \
  --tracker botsort \
  --botsort-with-reid \
  --track-thresh "$TRACK_THRESH" \
  --match-thresh "$MATCH_THRESH" \
  --track-buffer "$TRACK_BUFFER" \
  --min-track-len "$MIN_TRACK_LEN" \
  --state-priority-soft \
  --state-priority-iou-thresh "$STATE_IOU_THRESH" \
  --state-priority-score-gap-max "$STATE_SCORE_GAP_MAX" \
  --out "$OUT_PRELABELS"

echo "== 3/3 Convert to CVAT single-label JSON =="
python3 "$ROOT/scripts/datasets/coco_shift_category_ids_for_cvat.py" \
  --in "$OUT_PRELABELS" \
  --out "$OUT_CVAT" \
  --collapse-to-single-label ant \
  --carry-state-attributes

echo "Done."
echo "Prelabels: $OUT_PRELABELS"
echo "CVAT JSON: $OUT_CVAT"
