#!/usr/bin/env python3
"""RF-DETR inference → same COCO list JSON contract as infer_yolo.py (stub until wired)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_INF = Path(__file__).resolve().parent
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "RF-DETR (or other DETR-family) inference → COCO results JSON for evaluate.py. "
            "This entrypoint is a stub: implement model loading and loop below."
        )
    )
    p.add_argument("--weights", type=str, required=True, help="Checkpoint / weights path")
    p.add_argument("--source", type=str, required=True, help="Image file or directory")
    p.add_argument(
        "--coco-gt",
        type=str,
        required=True,
        help="COCO GT JSON (e.g. instances_val.json) to map file_name → image_id",
    )
    p.add_argument("--out", type=str, default="predictions.json", help="Output JSON path")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device string, e.g. "cuda:0" or "cpu".',
    )
    args = p.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    source_path = Path(args.source).expanduser().resolve()
    gt_path = Path(args.coco_gt).expanduser().resolve()

    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)
    if not source_path.exists():
        print(f"Source does not exist: {source_path}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.is_file():
        print(f"COCO GT not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    print(
        "infer_detr.py is not implemented yet.\n\n"
        "Output contract (match infer_yolo.py):\n"
        "  JSON list of dicts with keys: image_id (int), category_id (int), "
        "bbox [x, y, w, h] in absolute pixels, score (float).\n"
        "Use coco_pred_common.load_gt_filename_to_image_id(gt_path) for name → id.\n"
        "Use coco_pred_common.write_coco_predictions_json(path, detections) to write.\n\n"
        "Then run:\n"
        "  python scripts/evaluation/evaluate.py --gt <instances_val.json> --pred <out> ...\n"
        "  python scripts/evaluation/compare_metrics.py --baseline ... --compare ...\n",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
