#!/usr/bin/env python3
"""Run only viz_coco_overlays.run_comparisons (TP/FN/FP panels) to a chosen output directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

import viz_coco_overlays as vco  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", type=str, required=True)
    p.add_argument("--gt", type=str, required=True)
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True, help="Writes comparison PNGs here (flat).")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--score-thr", type=float, default=0.25)
    args = p.parse_args()

    pred_path = Path(args.pred).expanduser().resolve()
    gt_path = Path(args.gt).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not pred_path.is_file():
        print(f"Predictions not found: {pred_path}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.is_file():
        print(f"GT not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"Images dir not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    coco = vco._load_coco(gt_path)
    preds = vco._load_predictions(pred_path)
    labels = vco._category_names(coco)
    n = vco.run_comparisons(
        coco, preds, images_dir, out_dir, labels, args.score_thr, args.max_images
    )
    print(f"Wrote {n} comparison images → {out_dir}")


if __name__ == "__main__":
    main()
