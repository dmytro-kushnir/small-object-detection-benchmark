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

from coco_pred_common import (
    load_gt_filename_to_image_id,
    max_image_id_in_coco,
    write_coco_predictions_json,
)
from infer_image_common import dedupe_paths_preserve_order, iter_image_paths


def _append_yolo_detections(
    detections: list[dict],
    *,
    image_id: int,
    r: object,
) -> None:
    if r.boxes is None or len(r.boxes) == 0:
        return
    for b in r.boxes:
        xyxy = b.xyxy[0].tolist()
        x1, y1, x2, y2 = map(float, xyxy)
        cls = int(b.cls[0]) if b.cls is not None else 0
        score = float(b.conf[0]) if b.conf is not None else 1.0
        detections.append(
            {
                "image_id": image_id,
                "category_id": cls,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            }
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "YOLO inference → COCO results JSON (list) for pycocotools COCO.loadRes. "
            "Omit --coco-gt for external-only images (synthetic image_id). "
            "evaluate.py aligns predictions only when image_id matches GT images."
        )
    )
    p.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    p.add_argument("--source", type=str, required=True, help="Image file or directory")
    p.add_argument(
        "--coco-gt",
        type=str,
        default=None,
        help="COCO GT JSON for file_name → image_id (optional; omit for external-only mode)",
    )
    p.add_argument("--out", type=str, default="predictions.json", help="Output JSON path")
    p.add_argument(
        "--synthetic-image-id-start",
        type=int,
        default=1,
        help="First image_id when --coco-gt is omitted (sequential IDs in sorted path order)",
    )
    p.add_argument(
        "--extra-source",
        type=str,
        action="append",
        default=[],
        help="Extra image file or directory (repeatable); requires --coco-gt",
    )
    p.add_argument(
        "--extra-image-id-start",
        type=int,
        default=None,
        help="First image_id for --extra-source rows (default: max id in COCO + 1)",
    )
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

    if args.extra_source and not args.coco_gt:
        print("--extra-source requires --coco-gt", file=sys.stderr)
        sys.exit(1)

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
    detections: list[dict] = []
    missing_names: set[str] = set()

    if args.coco_gt is None:
        paths = iter_image_paths(source_path)
        if not paths:
            print("No images found under --source (check path and suffix).", file=sys.stderr)
            sys.exit(1)
        start = int(args.synthetic_image_id_start)
        id_by_resolve = {p: start + i for i, p in enumerate(paths)}
        results = model.predict(source=[str(p) for p in paths], **pred_kw)
        if args.save_vis:
            vp = Path(pred_kw["project"]) / pred_kw["name"]
            print(f"Saved visualizations under {vp}")
        for r in results:
            path_str = Path(r.path).as_posix() if r.path else ""
            if not path_str:
                continue
            key = Path(path_str).resolve()
            im_id = id_by_resolve.get(key)
            if im_id is None:
                continue
            _append_yolo_detections(detections, image_id=im_id, r=r)
    else:
        gt_path = Path(args.coco_gt).expanduser().resolve()
        if not gt_path.is_file():
            print(f"COCO GT not found: {gt_path}", file=sys.stderr)
            sys.exit(1)
        name_to_id = load_gt_filename_to_image_id(gt_path)

        results = model.predict(source=str(source_path), **pred_kw)
        if args.save_vis:
            vp = Path(pred_kw["project"]) / pred_kw["name"]
            print(f"Saved visualizations under {vp}")

        predicted_resolve: set[Path] = set()
        for r in results:
            path_str = Path(r.path).as_posix() if r.path else ""
            if not path_str:
                continue
            predicted_resolve.add(Path(path_str).resolve())
            base = Path(path_str).name
            if base not in name_to_id:
                missing_names.add(base)
                continue
            im_id = name_to_id[base]
            _append_yolo_detections(detections, image_id=im_id, r=r)

        if missing_names:
            print(
                f"Warning: {len(missing_names)} image file(s) not in COCO GT (skipped): "
                f"{sorted(missing_names)[:5]}{'…' if len(missing_names) > 5 else ''}",
                file=sys.stderr,
            )

        if args.extra_source:
            extra_all: list[Path] = []
            for es in args.extra_source:
                ep = Path(es).expanduser().resolve()
                if not ep.exists():
                    print(f"Extra source not found: {ep}", file=sys.stderr)
                    sys.exit(1)
                extra_all.extend(iter_image_paths(ep))
            extra_paths = dedupe_paths_preserve_order(extra_all)
            extra_paths = [p for p in extra_paths if p not in predicted_resolve]
            if extra_paths:
                x_start = (
                    int(args.extra_image_id_start)
                    if args.extra_image_id_start is not None
                    else max_image_id_in_coco(gt_path) + 1
                )
                id_by_resolve = {p: x_start + i for i, p in enumerate(extra_paths)}
                results_x = model.predict(source=[str(p) for p in extra_paths], **pred_kw)
                for r in results_x:
                    path_str = Path(r.path).as_posix() if r.path else ""
                    if not path_str:
                        continue
                    key = Path(path_str).resolve()
                    im_id = id_by_resolve.get(key)
                    if im_id is None:
                        continue
                    _append_yolo_detections(detections, image_id=im_id, r=r)

    out_path = Path(args.out).expanduser().resolve()
    write_coco_predictions_json(out_path, detections)
    print(f"Wrote {out_path} ({len(detections)} detections)")


if __name__ == "__main__":
    main()
