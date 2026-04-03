#!/usr/bin/env bash
# YOLO video tracking sweep: temporal-state-window K ∈ {0,3,5,9,15} for OOD / qualitative comparison.
# Requires: VIDEO=/path/to/clip.mp4
# Optional: WEIGHTS, OUT_DIR, CONF, IMGSZ, TRACKER, extra track_* args via env are not forwarded — edit script if needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VIDEO="${VIDEO:?Set VIDEO=/path/to/video.mp4}"
WEIGHTS="${WEIGHTS:-$ROOT/experiments/yolo/camponotus_idea1_trackidmajor_full_896/weights/best.pt}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/visualizations/ood_temporal_sweep}"
CONF="${CONF:-0.25}"
IMGSZ="${IMGSZ:-896}"

mkdir -p "$OUT_DIR"
STEM=$(basename "$VIDEO")
STEM="${STEM%.*}"

for K in 0 3 5 9 15; do
  OUTV="$OUT_DIR/${STEM}_yolo_k${K}.mp4"
  ANA="$OUT_DIR/${STEM}_yolo_k${K}_analytics.json"
  cmd=(
    python3 scripts/inference/track_yolo_video.py
    --weights "$WEIGHTS"
    --source-video "$VIDEO"
    --out-video "$OUTV"
    --imgsz "$IMGSZ"
    --conf "$CONF"
    --state-priority-soft
    --analytics-out "$ANA"
  )
  if [ "$K" -gt 0 ]; then
    cmd+=(--temporal-state-window "$K")
  fi
  echo "=== K=${K} -> ${ANA}"
  "${cmd[@]}"
done

echo "Done. Compare states/per_track in: $OUT_DIR/${STEM}_yolo_k*_analytics.json"
