#!/usr/bin/env python3
"""RF-DETR inference → COCO list JSON (evaluate.py compatible)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2

_INF = Path(__file__).resolve().parent
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

from coco_pred_common import (
    load_gt_filename_to_image_id,
    max_image_id_in_coco,
    write_coco_predictions_json,
)
from infer_image_common import (
    dedupe_paths_preserve_order,
    draw_coco_detection_records_on_bgr,
    filter_kwargs,
    iter_image_paths,
    rfdetr_output_to_coco_records,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "RF-DETR inference → COCO list JSON. "
            "Omit --coco-gt for external-only images (synthetic image_id). "
            "evaluate.py aligns predictions only when image_id matches GT images."
        )
    )
    p.add_argument("--weights", type=str, required=True, help="RF-DETR checkpoint .pth")
    p.add_argument("--source", type=str, required=True, help="Image file or directory")
    p.add_argument(
        "--coco-gt",
        type=str,
        default=None,
        help="COCO GT for file_name → image_id (optional; omit for external-only mode)",
    )
    p.add_argument("--out", type=str, required=True, help="Output predictions JSON")
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
        "--model-class",
        type=str,
        default="RFDETRSmall",
        help="rfdetr model class (must match training)",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument(
        "--class-id-mode",
        type=str,
        choices=("single", "multiclass"),
        default="single",
        help=(
            "single: map all detections to --category-id (default 0, ants). "
            "multiclass: use RF-DETR class indices as COCO category_id (e.g. Camponotus 0/1)."
        ),
    )
    p.add_argument(
        "--category-id",
        type=int,
        default=0,
        help="COCO category_id when --class-id-mode=single",
    )
    p.add_argument("--device", type=str, default=None, help="Optional torch device string")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument(
        "--save-vis",
        action="store_true",
        help="Save annotated images under --vis-dir (OpenCV overlay; same boxes as JSON).",
    )
    p.add_argument(
        "--vis-dir",
        type=str,
        default=None,
        help="Visualization output directory (default: parent of --out / viz_infer_rfdetr).",
    )
    p.add_argument(
        "--vis-names-json",
        type=str,
        default=None,
        help='Optional JSON object mapping category_id to label, e.g. {"0":"normal","1":"trophallaxis"}.',
    )
    args = p.parse_args()

    if args.extra_source and not args.coco_gt:
        print("--extra-source requires --coco-gt", file=sys.stderr)
        sys.exit(1)

    try:
        rfdetr = importlib.import_module("rfdetr")
    except ImportError:
        print("Install rfdetr (pip install rfdetr).", file=sys.stderr)
        sys.exit(1)

    if not hasattr(rfdetr, args.model_class):
        print(f"Unknown model class {args.model_class}", file=sys.stderr)
        sys.exit(1)
    ModelCls = getattr(rfdetr, args.model_class)

    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)
    if not source.exists():
        print(f"Source not found: {source}", file=sys.stderr)
        sys.exit(1)

    model = ModelCls(**filter_kwargs(ModelCls.__init__, {"pretrain_weights": str(weights)}))
    if args.device:
        dev = args.device.strip()
        if dev.isdigit():
            dev = f"cuda:{dev}"
        to_dev = getattr(model, "to", None)
        if callable(to_dev):
            try:
                import torch

                to_dev(torch.device(dev))
            except Exception:
                pass

    opt_flag = os.environ.get("EXP_A005_OPTIMIZE_INFERENCE", "0").strip().lower()
    if opt_flag in {"1", "true", "yes", "on"}:
        opt_fn = getattr(model, "optimize_for_inference", None)
        if callable(opt_fn):
            opt_fn()

    work: list[tuple[Path, int]] = []

    if args.coco_gt is None:
        paths = iter_image_paths(source)
        if not paths:
            print("No images found under --source (check path and suffix).", file=sys.stderr)
            sys.exit(1)
        start = int(args.synthetic_image_id_start)
        work = [(p, start + i) for i, p in enumerate(paths)]
    else:
        gt_path = Path(args.coco_gt).expanduser().resolve()
        if not gt_path.is_file():
            print(f"COCO GT not found: {gt_path}", file=sys.stderr)
            sys.exit(1)
        name_to_id = load_gt_filename_to_image_id(gt_path)
        coco = json.loads(gt_path.read_text(encoding="utf-8"))
        work_gt: list[tuple[Path, int]] = []

        if source.is_file():
            base = source.name
            if base in name_to_id:
                work_gt.append((source.resolve(), int(name_to_id[base])))
        elif source.is_dir():
            for im in coco.get("images", []):
                fn = im.get("file_name")
                if fn is None:
                    continue
                bname = Path(str(fn)).name
                if bname not in name_to_id:
                    continue
                ip = source / bname
                if ip.is_file():
                    work_gt.append((ip.resolve(), int(name_to_id[bname])))
        else:
            print(f"Source must be a file or directory: {source}", file=sys.stderr)
            sys.exit(1)

        if args.extra_source:
            gt_resolves = {p for p, _ in work_gt}
            extra_all: list[Path] = []
            for es in args.extra_source:
                ep = Path(es).expanduser().resolve()
                if not ep.exists():
                    print(f"Extra source not found: {ep}", file=sys.stderr)
                    sys.exit(1)
                extra_all.extend(iter_image_paths(ep))
            extra_paths = dedupe_paths_preserve_order(extra_all)
            extra_paths = [p for p in extra_paths if p not in gt_resolves]
            if extra_paths:
                x_start = (
                    int(args.extra_image_id_start)
                    if args.extra_image_id_start is not None
                    else max_image_id_in_coco(gt_path) + 1
                )
                work_extra = [(p, x_start + i) for i, p in enumerate(extra_paths)]
                work_gt.extend(work_extra)

        work = work_gt

    if args.max_images is not None:
        work = work[: max(0, int(args.max_images))]

    vis_dir: Path | None = None
    class_names: dict[int, str] | None = None
    if args.save_vis:
        out_parent = out_path.parent
        vis_dir = (
            Path(args.vis_dir).expanduser().resolve()
            if args.vis_dir
            else out_parent / "viz_infer_rfdetr"
        )
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.vis_names_json:
        raw_names = json.loads(Path(args.vis_names_json).expanduser().read_text(encoding="utf-8"))
        if not isinstance(raw_names, dict):
            print("--vis-names-json must be a JSON object", file=sys.stderr)
            sys.exit(1)
        class_names = {}
        for k, v in raw_names.items():
            try:
                class_names[int(k)] = str(v)
            except (TypeError, ValueError):
                continue

    detections: list[dict[str, Any]] = []
    predict_kw = filter_kwargs(
        model.predict,
        {"threshold": float(args.conf), "confidence": float(args.conf)},
    )
    for path, im_id in work:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw = model.predict(rgb, **predict_kw)
        recs = rfdetr_output_to_coco_records(
            raw,
            im_id,
            category_id_fallback=int(args.category_id),
            class_id_mode=str(args.class_id_mode),
        )
        detections.extend(recs)
        if vis_dir is not None:
            vis_bgr = draw_coco_detection_records_on_bgr(
                bgr, recs, class_names=class_names
            )
            cv2.imwrite(str(vis_dir / path.name), vis_bgr)

    write_coco_predictions_json(out_path, detections)
    print(f"Wrote {out_path} ({len(detections)} detections)")
    if vis_dir is not None:
        print(f"Saved visualizations under {vis_dir}")


if __name__ == "__main__":
    main()
