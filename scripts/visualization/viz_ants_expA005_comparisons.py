#!/usr/bin/env python3
"""YOLO vs RF-DETR comparison panels: per-model TP/FN/FP viz + optional side-by-side strip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))

import viz_coco_overlays as vco  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=str, required=True)
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--pred-yolo", type=str, required=True)
    p.add_argument("--pred-rfdetr", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--max-images", type=int, default=250)
    p.add_argument("--score-thr", type=float, default=0.25)
    args = p.parse_args()

    gt_path = Path(args.gt).expanduser().resolve()
    img_dir = Path(args.images_dir).expanduser().resolve()
    py = Path(args.pred_yolo).expanduser().resolve()
    pr = Path(args.pred_rfdetr).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()

    for f in (gt_path, py, pr, img_dir):
        if not f.exists():
            print(f"Missing path: {f}", file=sys.stderr)
            sys.exit(1)

    coco = vco._load_coco(gt_path)
    labels = vco._category_names(coco)
    preds_y = vco._load_predictions(py)
    preds_r = vco._load_predictions(pr)

    dir_yolo = out_root / "yolo_panels"
    dir_rf = out_root / "rfdetr_panels"
    dir_side = out_root / "side_by_side"

    n_y = vco.run_comparisons(
        coco, preds_y, img_dir, dir_yolo, labels, args.score_thr, args.max_images
    )
    n_r = vco.run_comparisons(
        coco, preds_r, img_dir, dir_rf, labels, args.score_thr, args.max_images
    )

    dir_side.mkdir(parents=True, exist_ok=True)
    n_ss = 0
    for pimg in sorted(dir_yolo.glob("*.jpg")):
        q = dir_rf / pimg.name
        if not q.is_file():
            continue
        a = cv2.imread(str(pimg))
        b = cv2.imread(str(q))
        if a is None or b is None:
            continue
        h = min(a.shape[0], b.shape[0])
        if a.shape[0] != h:
            a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
        if b.shape[0] != h:
            b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
        gap = np.zeros((h, 8, 3), dtype=np.uint8)
        lab_y = np.zeros((32, a.shape[1] + b.shape[1] + 8, 3), dtype=np.uint8)
        lab_y[:] = (40, 40, 40)
        cv2.putText(lab_y, "YOLO768", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(
            lab_y,
            "RF-DETR",
            (a.shape[1] + 12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
        )
        row = np.hstack([a, gap, b])
        combo = np.vstack([lab_y, row])
        cv2.imwrite(str(dir_side / pimg.name), combo)
        n_ss += 1

    print(f"Wrote {n_y} YOLO panels → {dir_yolo}")
    print(f"Wrote {n_r} RF-DETR panels → {dir_rf}")
    print(f"Wrote {n_ss} side-by-side → {dir_side}")


if __name__ == "__main__":
    main()
