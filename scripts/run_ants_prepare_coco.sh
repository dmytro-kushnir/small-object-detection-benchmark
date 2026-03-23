#!/usr/bin/env bash
# Build datasets/ants_coco/ for RF-DETR from datasets/ants_yolo/ (same split).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CFG="${ANTS_COCO_PREP_CONFIG:-$ROOT/configs/datasets/ants_coco_rfdetr.yaml}"
python3 "$ROOT/scripts/datasets/prepare_ants_coco_rfdetr.py" --config "$CFG"
