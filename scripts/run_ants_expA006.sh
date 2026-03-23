#!/usr/bin/env bash
# EXP-A006: RF-DETR optimized + ByteTrack + temporal smoothing on ants val.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG="${EXP_A006_CONFIG:-$ROOT/configs/expA006_ants_tracking.yaml}"
if [[ ! -f "$CFG" ]]; then
  echo "Missing config: $CFG" >&2
  exit 1
fi

read_cfg() {
  python3 - "$CFG" "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8')) or {}
key = sys.argv[2]
val = cfg.get(key)
print("" if val is None else val)
PY
}

GT_VAL="${EXP_A006_GT_VAL:-$ROOT/$(read_cfg rfdetr_gt_val)}"
IMG_VAL="${EXP_A006_IMG_VAL:-$ROOT/$(read_cfg rfdetr_images_val_dir)}"
MODEL_CLASS="${EXP_A006_MODEL_CLASS:-$(read_cfg rfdetr_model_class)}"
CONF_THR="${EXP_A006_CONF_THR:-$(read_cfg rfdetr_conf_threshold)}"
WEIGHTS="${EXP_A006_WEIGHTS:-$ROOT/$(read_cfg rfdetr_weights)}"
PRED_OPT="${EXP_A006_PRED_OPT:-$ROOT/$(read_cfg rfdetr_pred_optimized)}"
DET_BENCH="${EXP_A006_DET_BENCH:-$ROOT/$(read_cfg rfdetr_bench_optimized)}"
TRAIN_CFG="${EXP_A006_TRAIN_CFG:-$ROOT/$(read_cfg rfdetr_train_config)}"
MANIFEST="${EXP_A006_MANIFEST:-$ROOT/$(read_cfg sequence_manifest)}"

TRACKS_OUT="${EXP_A006_TRACKS_OUT:-$ROOT/$(read_cfg tracks_out)}"
TRACK_STATS="${EXP_A006_TRACK_STATS_OUT:-$ROOT/$(read_cfg tracking_stats_out)}"
SM_TRACKS_OUT="${EXP_A006_SMOOTHED_TRACKS_OUT:-$ROOT/$(read_cfg smoothed_tracks_out)}"
SM_PRED_OUT="${EXP_A006_SMOOTHED_PRED_OUT:-$ROOT/$(read_cfg smoothed_predictions_out)}"
SM_STATS="${EXP_A006_SMOOTHING_STATS_OUT:-$ROOT/$(read_cfg smoothing_stats_out)}"
PIPE_BENCH="${EXP_A006_PIPELINE_BENCH_OUT:-$ROOT/$(read_cfg pipeline_benchmark_out)}"
CFG_OUT="${EXP_A006_CONFIG_OUT:-$ROOT/$(read_cfg config_out)}"
SYS_OUT="${EXP_A006_SYSTEM_INFO_OUT:-$ROOT/$(read_cfg system_info_out)}"

MET_OUT="${EXP_A006_METRICS_OUT:-$ROOT/$(read_cfg metrics_out)}"
BASE_MET="${EXP_A006_BASELINE_METRICS:-$ROOT/$(read_cfg baseline_metrics)}"
CMP_OUT="${EXP_A006_COMPARE_OUT:-$ROOT/$(read_cfg compare_out)}"
VIZ_DIR="${EXP_A006_VIZ_DIR:-$ROOT/$(read_cfg viz_out_dir)}"
SUMMARY_OUT="${EXP_A006_SUMMARY_OUT:-$ROOT/$(read_cfg summary_out)}"
EXP_ID="${EXP_A006_EXPERIMENT_ID:-$(read_cfg experiment_id)}"

MIN_TRACK_LEN="${EXP_A006_MIN_TRACK_LEN:-$(read_cfg min_track_len)}"
FILL_GAP_MAX="${EXP_A006_FILL_GAP_MAX:-$(read_cfg fill_gap_max)}"
TRACK_THRESH="${EXP_A006_TRACK_THRESH:-$(read_cfg track_thresh)}"
MATCH_THRESH="${EXP_A006_MATCH_THRESH:-$(read_cfg match_thresh)}"
TRACK_BUFFER="${EXP_A006_TRACK_BUFFER:-$(read_cfg track_buffer)}"
SEG_FILTER_DISTANCE_ABS="${EXP_A006_SEG_FILTER_DISTANCE_ABS:-$(read_cfg seg_filter_distance_abs)}"
SEG_FILTER_DISTANCE_RATIO="${EXP_A006_SEG_FILTER_DISTANCE_RATIO:-$(read_cfg seg_filter_distance_ratio)}"
REUSE_PRED="${EXP_A006_REUSE_PREDICTIONS:-$(read_cfg reuse_predictions)}"

DEVICE="${RFDETR_DEVICE:-${ANTS_DEVICE:-${SMOKE_DEVICE:-auto}}}"
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python3 -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo cpu)"
fi

mkdir -p "$(dirname "$TRACKS_OUT")" "$(dirname "$MET_OUT")" "$(dirname "$VIZ_DIR")"
cp "$CFG" "$CFG_OUT"

python3 - <<'PY' "$SYS_OUT"
import json, platform, sys
from pathlib import Path
try:
    import torch
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception:
    info = {"python": sys.version, "platform": platform.platform()}
p = Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(info, indent=2), encoding='utf-8')
print(f"Wrote {p}")
PY

if [[ ! -f "$PRED_OPT" || "${REUSE_PRED,,}" != "true" ]]; then
  echo "== Run optimized RF-DETR inference for EXP-A006 =="
  export EXP_A005_OPTIMIZE_INFERENCE=1
  python3 "$ROOT/scripts/inference/infer_rfdetr.py" \
    --weights "$WEIGHTS" \
    --source "$IMG_VAL" \
    --coco-gt "$GT_VAL" \
    --out "$PRED_OPT" \
    --model-class "$MODEL_CLASS" \
    --conf "$CONF_THR" \
    --device "$DEVICE"
fi

if [[ ! -f "$DET_BENCH" ]]; then
  echo "== Run optimized RF-DETR benchmark for EXP-A006 =="
  export EXP_A005_OPTIMIZE_INFERENCE=1
  python3 "$ROOT/scripts/evaluation/bench_rfdetr.py" \
    --weights "$WEIGHTS" \
    --source "$IMG_VAL" \
    --coco-gt "$GT_VAL" \
    --model-class "$MODEL_CLASS" \
    --conf "$CONF_THR" \
    --device "$DEVICE" \
    --out "$DET_BENCH" \
    --config "$CFG"
fi

echo "== ByteTrack tracking =="
MANIFEST_ARGS=()
if [[ -n "$MANIFEST" && -f "$MANIFEST" ]]; then
  MANIFEST_ARGS+=(--manifest "$MANIFEST")
fi
python3 "$ROOT/scripts/inference/track_rfdetr_bytetrack.py" \
  --gt "$GT_VAL" \
  --pred "$PRED_OPT" \
  "${MANIFEST_ARGS[@]}" \
  --out "$TRACKS_OUT" \
  --stats-out "$TRACK_STATS" \
  --track-thresh "$TRACK_THRESH" \
  --match-thresh "$MATCH_THRESH" \
  --track-buffer "$TRACK_BUFFER" \
  ${SEG_FILTER_DISTANCE_ABS:+--seg-filter-distance-abs "$SEG_FILTER_DISTANCE_ABS"} \
  ${SEG_FILTER_DISTANCE_RATIO:+--seg-filter-distance-ratio "$SEG_FILTER_DISTANCE_RATIO"}

echo "== Temporal smoothing =="
python3 "$ROOT/scripts/inference/smooth_tracks_expA006.py" \
  --gt "$GT_VAL" \
  --tracks "$TRACKS_OUT" \
  "${MANIFEST_ARGS[@]}" \
  --out-pred "$SM_PRED_OUT" \
  --out-tracks "$SM_TRACKS_OUT" \
  --stats-out "$SM_STATS" \
  --min-track-len "$MIN_TRACK_LEN" \
  --fill-gap-max "$FILL_GAP_MAX"

echo "== Compose pipeline benchmark =="
python3 "$ROOT/scripts/evaluation/bench_expA006_tracking.py" \
  --detector-bench "$DET_BENCH" \
  --tracking-stats "$TRACK_STATS" \
  --smoothing-stats "$SM_STATS" \
  --out "$PIPE_BENCH"

echo "== Evaluate smoothed predictions =="
EVAL_EXTRA=()
if [[ -f "$MANIFEST" ]]; then
  EVAL_EXTRA+=(--prepare-manifest "$MANIFEST")
fi
python3 "$ROOT/scripts/evaluation/evaluate.py" \
  --gt "$GT_VAL" \
  --pred "$SM_PRED_OUT" \
  --weights "$WEIGHTS" \
  --images-dir "$IMG_VAL" \
  --out "$MET_OUT" \
  --experiment-id "$EXP_ID" \
  --train-config "$TRAIN_CFG" \
  "${EVAL_EXTRA[@]}" \
  --device "$DEVICE" \
  --inference-benchmark-json "$PIPE_BENCH"

echo "== Compare vs EXP-A005 optimized baseline =="
python3 "$ROOT/scripts/evaluation/compare_ants_expA006.py" \
  --baseline "$BASE_MET" \
  --compare "$MET_OUT" \
  --out "$CMP_OUT"

echo "== Visualize tracking and smoothing =="
python3 "$ROOT/scripts/visualization/viz_ants_expA006_tracking.py" \
  --gt "$GT_VAL" \
  --images-dir "$IMG_VAL" \
  --pred-before "$PRED_OPT" \
  --pred-after "$SM_PRED_OUT" \
  --tracks "$SM_TRACKS_OUT" \
  --out-dir "$VIZ_DIR" \
  --max-images "${EXP_A006_VIZ_MAX_IMAGES:-250}"

echo "== Write EXP-A006 summary =="
python3 "$ROOT/scripts/evaluation/write_ants_expA006_summary.py" \
  --compare "$CMP_OUT" \
  --out "$SUMMARY_OUT"

echo "EXP-A006 finished."
