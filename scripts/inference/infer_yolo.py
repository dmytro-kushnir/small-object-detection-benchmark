#!/usr/bin/env python3
"""YOLO inference wrapper (stub): run Ultralytics predict and emit COCO-style JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO inference → COCO predictions JSON (stub).")
    p.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    p.add_argument("--source", type=str, required=True, help="Image or directory")
    p.add_argument("--out", type=str, default="predictions.json", help="Output JSON path")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Install ultralytics (see requirements.txt; use >=8.4.0 for YOLO26).",
            file=sys.stderr,
        )
        sys.exit(1)

    model = YOLO(args.weights)
    results = model.predict(source=args.source, save=False, verbose=False)

    preds: list[dict] = []
    for r in results:
        path = Path(r.path).as_posix() if r.path else ""
        if r.orig_shape is not None:
            h, w = int(r.orig_shape[0]), int(r.orig_shape[1])
        else:
            h, w = 0, 0
        im_id = abs(hash(path)) % (10**9)
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = map(float, xyxy)
            cls = int(b.cls[0]) if b.cls is not None else 0
            score = float(b.conf[0]) if b.conf is not None else 1.0
            preds.append(
                {
                    "image_id": im_id,
                    "file_name": path,
                    "category_id": cls,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                    "image_width": w,
                    "image_height": h,
                }
            )

    out = {"annotations": preds, "info": {"description": "Ultralytics predict (stub schema)"}}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
