#!/usr/bin/env python3
"""RF-DETR inference → COCO list JSON (evaluate.py compatible)."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {k: v for k, v in kwargs.items() if v is not None}
    return {k: v for k, v in kwargs.items() if k in params and v is not None}

import cv2
import numpy as np

_INF = Path(__file__).resolve().parent
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

from coco_pred_common import load_gt_filename_to_image_id, write_coco_predictions_json


def _detections_to_records(
    det_out: Any,
    image_id: int,
    category_id_fallback: int = 0,
) -> list[dict[str, Any]]:
    """Convert rfdetr/supervision Detections (or similar) to COCO detection dicts."""
    if det_out is None:
        return []
    d = det_out
    if isinstance(d, (list, tuple)) and len(d) > 0:
        d = d[0]

    xyxy = getattr(d, "xyxy", None)
    if xyxy is None:
        return []
    xyxy = np.asarray(xyxy)
    if xyxy.size == 0:
        return []

    n = xyxy.shape[0]
    cls = getattr(d, "class_id", None)
    if cls is not None:
        cls = np.asarray(cls).reshape(-1)
    else:
        cls = np.zeros(n, dtype=np.int64)

    conf = getattr(d, "confidence", None)
    if conf is not None:
        conf = np.asarray(conf).reshape(-1)
    else:
        conf = np.ones(n, dtype=np.float64)

    records: list[dict[str, Any]] = []
    for i in range(n):
        x1, y1, x2, y2 = map(float, xyxy[i].tolist())
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        cid = int(cls[i]) if i < len(cls) else category_id_fallback
        # Single-class ants: map any class index to COCO category 0
        cid = category_id_fallback
        sc = float(conf[i]) if i < len(conf) else 1.0
        records.append(
            {
                "image_id": int(image_id),
                "category_id": cid,
                "bbox": [x1, y1, w, h],
                "score": sc,
            }
        )
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True, help="RF-DETR checkpoint .pth")
    p.add_argument("--source", type=str, required=True, help="Val image directory")
    p.add_argument("--coco-gt", type=str, required=True, help="COCO GT for file_name → image_id")
    p.add_argument("--out", type=str, required=True, help="Output predictions JSON")
    p.add_argument(
        "--model-class",
        type=str,
        default="RFDETRSmall",
        help="rfdetr model class (must match training)",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--device", type=str, default=None, help="Optional torch device string")
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

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
    gt_path = Path(args.coco_gt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)
    if not source.is_dir():
        print(f"Source not found: {source}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.is_file():
        print(f"COCO GT not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    model = ModelCls(**_filter_kwargs(ModelCls.__init__, {"pretrain_weights": str(weights)}))
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

    # Optional inference optimization (used for the dedicated "optimized inference" run).
    # This is environment-controlled to keep the unoptimized baseline reproducible.
    opt_flag = os.environ.get("EXP_A005_OPTIMIZE_INFERENCE", "0").strip().lower()
    if opt_flag in {"1", "true", "yes", "on"}:
        opt_fn = getattr(model, "optimize_for_inference", None)
        if callable(opt_fn):
            opt_fn()

    name_to_id = load_gt_filename_to_image_id(gt_path)
    import json

    coco = json.loads(gt_path.read_text(encoding="utf-8"))
    work: list[tuple[Path, int]] = []
    for im in coco.get("images", []):
        fn = im.get("file_name")
        iid = im.get("id")
        if fn is None or iid is None:
            continue
        base = Path(str(fn)).name
        if base not in name_to_id:
            continue
        ip = source / base
        if ip.is_file():
            work.append((ip, int(name_to_id[base])))

    if args.max_images is not None:
        work = work[: max(0, args.max_images)]

    detections: list[dict[str, Any]] = []
    predict_kw = _filter_kwargs(
        model.predict,
        {"threshold": float(args.conf), "confidence": float(args.conf)},
    )
    for path, im_id in work:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw = model.predict(rgb, **predict_kw)
        detections.extend(_detections_to_records(raw, im_id, category_id_fallback=0))

    write_coco_predictions_json(out_path, detections)
    print(f"Wrote {out_path} ({len(detections)} detections)")


if __name__ == "__main__":
    main()
