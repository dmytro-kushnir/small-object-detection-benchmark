#!/usr/bin/env python3
"""YOLO inference: Ultralytics predict → COCO detection results (pycocotools-compatible list JSON)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Same directory as this script (supports `python scripts/inference/infer_yolo.py`).
_INF = Path(__file__).resolve().parent
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

from coco_pred_common import load_gt_filename_to_image_id, write_coco_predictions_json


def main() -> None:
    p = argparse.ArgumentParser(
        description="YOLO inference → COCO results JSON (list) for pycocotools COCO.loadRes."
    )
    p.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
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
        help='Ultralytics device: "0", "cpu", etc. Default: Ultralytics auto.',
    )
    p.add_argument(
        "--save-vis",
        action="store_true",
        help="Save annotated images under --vis-dir (Ultralytics predict).",
    )
    p.add_argument(
        "--vis-dir",
        type=str,
        default=None,
        help="Visualization folder (parent=project, basename=name for Ultralytics).",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Ultralytics predict imgsz (default: model/YOLO default).",
    )
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Install ultralytics (see requirements.txt; use >=8.4.0 for YOLO26).",
            file=sys.stderr,
        )
        sys.exit(1)

    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)
    source_path = Path(args.source).expanduser().resolve()
    if not source_path.exists():
        print(f"Source does not exist: {source_path}", file=sys.stderr)
        sys.exit(1)
    gt_path = Path(args.coco_gt).expanduser().resolve()
    if not gt_path.is_file():
        print(f"COCO GT not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    name_to_id = load_gt_filename_to_image_id(gt_path)

    pred_kw: dict = {"save": False, "verbose": False}
    if args.device is not None:
        pred_kw["device"] = args.device
    if args.imgsz is not None:
        pred_kw["imgsz"] = int(args.imgsz)
    if args.save_vis:
        out_parent = Path(args.out).expanduser().resolve().parent
        vis_path = (
            Path(args.vis_dir).expanduser().resolve()
            if args.vis_dir
            else out_parent / "viz_infer"
        )
        pred_kw["save"] = True
        pred_kw["project"] = str(vis_path.parent)
        pred_kw["name"] = vis_path.name

    model = YOLO(str(weights))
    results = model.predict(source=str(source_path), **pred_kw)
    if args.save_vis:
        vp = Path(pred_kw["project"]) / pred_kw["name"]
        print(f"Saved visualizations under {vp}")

    detections: list[dict] = []
    missing_names: set[str] = set()

    for r in results:
        path_str = Path(r.path).as_posix() if r.path else ""
        if not path_str:
            continue
        base = Path(path_str).name
        if base not in name_to_id:
            missing_names.add(base)
            continue
        im_id = name_to_id[base]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = map(float, xyxy)
            cls = int(b.cls[0]) if b.cls is not None else 0
            score = float(b.conf[0]) if b.conf is not None else 1.0
            detections.append(
                {
                    "image_id": im_id,
                    "category_id": cls,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )

    if missing_names:
        print(
            f"Warning: {len(missing_names)} image file(s) not in COCO GT (skipped): "
            f"{sorted(missing_names)[:5]}{'…' if len(missing_names) > 5 else ''}",
            file=sys.stderr,
        )

    out_path = Path(args.out).expanduser().resolve()
    write_coco_predictions_json(out_path, detections)
    print(f"Wrote {out_path} ({len(detections)} detections)")


if __name__ == "__main__":
    main()
