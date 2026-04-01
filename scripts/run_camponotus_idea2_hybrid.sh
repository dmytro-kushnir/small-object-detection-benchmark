#!/usr/bin/env bash
# One-shot Idea 2 hybrid runner (MOT -> events -> eval -> compare).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- Config (override with env vars) ---------------------------------------
GT_EVENTS="${GT_EVENTS:-$ROOT/datasets/camponotus_idea2_event_benchmark_v1.json}"
IN_SITU_ROOT="${IN_SITU_ROOT:-/media/dmytro/data/datasets/camponotus fellah trophallaxis FULL dataset/images/default/in_situ}"
RUN_NAME="${RUN_NAME:-camponotus_idea2_hybrid_v1}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/results/$RUN_NAME}"

# Comma-separated sequence folder names under IN_SITU_ROOT.
SEQ_LIST="${SEQ_LIST:-seq_camponotus_trophallaxis_007,seq_camponotus_trophallaxis_005,seq_camponotus_trophallaxis_003,seq_camponotus_009,seq_camponotus_007,seq_camponotus_003,seq_camponotus_002,seq_camponotus_010}"

# Detector backend: yolo | rfdetr
BACKEND="${BACKEND:-yolo}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-$ROOT/experiments/yolo/camponotus_idea1_trackidmajor_full_896/weights/best.pt}"
YOLO_IMGSZ="${YOLO_IMGSZ:-896}"
RFDETR_WEIGHTS="${RFDETR_WEIGHTS:-}"
RFDETR_MODEL_CLASS="${RFDETR_MODEL_CLASS:-RFDETRSmall}"

# Tracking / thresholds.
CONF="${CONF:-0.25}"
TRACKER="${TRACKER:-botsort}"
TRACK_THRESH="${TRACK_THRESH:-0.25}"
MATCH_THRESH="${MATCH_THRESH:-0.8}"
TRACK_BUFFER="${TRACK_BUFFER:-30}"
MIN_TRACK_LEN="${MIN_TRACK_LEN:-2}"

# Idea 2 event inference thresholds.
MAX_DIST_PX="${MAX_DIST_PX:-90}"
PAIR_SCORE_THRESHOLD="${PAIR_SCORE_THRESHOLD:-0.45}"
MIN_ACTIVE_FRAMES="${MIN_ACTIVE_FRAMES:-12}"
MAX_GAP_FRAMES="${MAX_GAP_FRAMES:-3}"
MATCH_TIOU_THRESHOLD="${MATCH_TIOU_THRESHOLD:-0.30}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$GT_EVENTS" ]]; then
  echo "GT events file not found: $GT_EVENTS" >&2
  exit 1
fi
if [[ ! -d "$IN_SITU_ROOT" ]]; then
  echo "In-situ root not found: $IN_SITU_ROOT" >&2
  exit 1
fi

if [[ "$BACKEND" == "yolo" ]]; then
  if [[ ! -f "$YOLO_WEIGHTS" ]]; then
    echo "YOLO weights not found: $YOLO_WEIGHTS" >&2
    exit 1
  fi
elif [[ "$BACKEND" == "rfdetr" ]]; then
  if [[ ! -f "$RFDETR_WEIGHTS" ]]; then
    echo "RF-DETR weights not found: $RFDETR_WEIGHTS" >&2
    exit 1
  fi
else
  echo "Unsupported BACKEND='$BACKEND' (expected yolo or rfdetr)." >&2
  exit 1
fi

echo "== Stage selected sequences =="
STAGE_DIR="$(mktemp -d -t camponotus_idea2_stage_XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT

IFS=',' read -r -a SEQS <<< "$SEQ_LIST"
if [[ ${#SEQS[@]} -eq 0 ]]; then
  echo "SEQ_LIST is empty." >&2
  exit 1
fi
for seq in "${SEQS[@]}"; do
  seq_trimmed="$(echo "$seq" | xargs)"
  src="$IN_SITU_ROOT/$seq_trimmed"
  if [[ ! -d "$src" ]]; then
    echo "Missing sequence folder: $src" >&2
    exit 1
  fi
  ln -s "$src" "$STAGE_DIR/$seq_trimmed"
done

PRELABEL_COCO="$OUT_DIR/${RUN_NAME}_prelabels_coco.json"
MOT_JSON="$OUT_DIR/${RUN_NAME}_mot.json"
EVENTS_WITH_HELPER="$OUT_DIR/${RUN_NAME}_events_with_helper.json"
EVENTS_NO_HELPER="$OUT_DIR/${RUN_NAME}_events_no_helper.json"
EVAL_WITH_HELPER="$OUT_DIR/${RUN_NAME}_events_with_helper_eval.json"
EVAL_NO_HELPER="$OUT_DIR/${RUN_NAME}_events_no_helper_eval.json"
COMPARE_JSON="$OUT_DIR/${RUN_NAME}_events_helper_vs_no_helper.json"

echo "== Build MOT JSON from selected sequences =="
if [[ "$BACKEND" == "yolo" ]]; then
  python3 scripts/datasets/bootstrap_camponotus_autolabel.py \
    --images-root "$STAGE_DIR" \
    --backend yolo \
    --yolo-weights "$YOLO_WEIGHTS" \
    --yolo-imgsz "$YOLO_IMGSZ" \
    --conf "$CONF" \
    --with-tracking \
    --tracker "$TRACKER" \
    --track-thresh "$TRACK_THRESH" \
    --match-thresh "$MATCH_THRESH" \
    --track-buffer "$TRACK_BUFFER" \
    --min-track-len "$MIN_TRACK_LEN" \
    --out "$PRELABEL_COCO" \
    --mot-out-json "$MOT_JSON"
else
  python3 scripts/datasets/bootstrap_camponotus_autolabel.py \
    --images-root "$STAGE_DIR" \
    --backend rfdetr \
    --rfdetr-weights "$RFDETR_WEIGHTS" \
    --rfdetr-model-class "$RFDETR_MODEL_CLASS" \
    --conf "$CONF" \
    --with-tracking \
    --tracker "$TRACKER" \
    --track-thresh "$TRACK_THRESH" \
    --match-thresh "$MATCH_THRESH" \
    --track-buffer "$TRACK_BUFFER" \
    --min-track-len "$MIN_TRACK_LEN" \
    --out "$PRELABEL_COCO" \
    --mot-out-json "$MOT_JSON"
fi

echo "== Infer Idea 2 events (with helper) =="
python3 scripts/inference/infer_camponotus_idea2_events.py \
  --mot-json "$MOT_JSON" \
  --out "$EVENTS_WITH_HELPER" \
  --max-dist-px "$MAX_DIST_PX" \
  --pair-score-threshold "$PAIR_SCORE_THRESHOLD" \
  --min-active-frames "$MIN_ACTIVE_FRAMES" \
  --max-gap-frames "$MAX_GAP_FRAMES"

echo "== Infer Idea 2 events (no helper) =="
python3 scripts/inference/infer_camponotus_idea2_events.py \
  --mot-json "$MOT_JSON" \
  --out "$EVENTS_NO_HELPER" \
  --disable-helper-signal \
  --max-dist-px "$MAX_DIST_PX" \
  --pair-score-threshold "$PAIR_SCORE_THRESHOLD" \
  --min-active-frames "$MIN_ACTIVE_FRAMES" \
  --max-gap-frames "$MAX_GAP_FRAMES"

echo "== Evaluate with helper =="
python3 scripts/evaluation/evaluate_camponotus_idea2_events.py \
  --gt-events "$GT_EVENTS" \
  --pred-events "$EVENTS_WITH_HELPER" \
  --out "$EVAL_WITH_HELPER" \
  --match-tiou-threshold "$MATCH_TIOU_THRESHOLD"

echo "== Evaluate no helper =="
python3 scripts/evaluation/evaluate_camponotus_idea2_events.py \
  --gt-events "$GT_EVENTS" \
  --pred-events "$EVENTS_NO_HELPER" \
  --out "$EVAL_NO_HELPER" \
  --match-tiou-threshold "$MATCH_TIOU_THRESHOLD"

echo "== Compare helper vs no-helper =="
python3 scripts/evaluation/compare_camponotus_idea2_event_metrics.py \
  --baseline "$EVAL_NO_HELPER" \
  --compare "$EVAL_WITH_HELPER" \
  --out "$COMPARE_JSON" \
  --evaluation-note "Idea 2 one-shot run: helper signal contribution."

echo
echo "Done."
echo "Output directory: $OUT_DIR"
echo "MOT JSON: $MOT_JSON"
echo "With helper eval: $EVAL_WITH_HELPER"
echo "No helper eval: $EVAL_NO_HELPER"
echo "Compare bundle: $COMPARE_JSON"
