#!/usr/bin/env python3
"""Draw ROI rectangles from experiments/yolo/ants_expA004/rois.json (cyan) for debugging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rois-json",
        type=str,
        default="experiments/yolo/ants_expA004/rois.json",
    )
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument(
        "--out-dir",
        type=str,
        default="experiments/visualizations/ants_expA004/rois_debug",
    )
    p.add_argument("--max-images", type=int, default=50)
    args = p.parse_args()

    rois_path = Path(args.rois_json).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not rois_path.is_file():
        print(f"rois.json not found: {rois_path}", file=sys.stderr)
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"Images dir not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(rois_path.read_text(encoding="utf-8"))
    images: list[dict[str, Any]] = list(data.get("images") or [])
    if args.max_images is not None:
        images = images[: max(0, args.max_images)]

    cyan = (255, 255, 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for rec in images:
        fn = rec.get("file_name")
        if not fn:
            continue
        base = Path(str(fn)).name
        ip = images_dir / base
        if not ip.is_file():
            continue
        arr = cv2.imread(str(ip))
        if arr is None:
            continue
        for xyxy in rec.get("rois") or []:
            if not isinstance(xyxy, list) or len(xyxy) < 4:
                continue
            x1, y1, x2, y2 = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            cv2.rectangle(arr, (x1, y1), (x2, y2), cyan, 2)
        cv2.putText(
            arr,
            f"n_rois={rec.get('n_rois', 0)}",
            (4, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            cyan,
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / base), arr)
        n += 1
    print(f"Wrote {n} ROI debug images → {out_dir}")


if __name__ == "__main__":
    main()
