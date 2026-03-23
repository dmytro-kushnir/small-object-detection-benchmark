#!/usr/bin/env python3
"""Generate model prelabels (COCO detections) for Camponotus annotation bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

from camponotus_common import CAMPO_CLASSES, write_json


def _collect_images(images_root: Path, max_images: int | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = sorted(p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if max_images is not None:
        imgs = imgs[: max(0, max_images)]
    return imgs


def _infer_with_yolo(weights: Path, image: Any, conf: float) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict(image, conf=conf, verbose=False)
    out: list[dict[str, Any]] = []
    if not results:
        return out
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy()
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
        out.append(
            {
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(confs[i]),
                "category_id": int(clss[i]) if int(clss[i]) in (0, 1) else 0,
            }
        )
    return out


def _infer_with_rfdetr(weights: Path, image: Any, conf: float, model_class: str) -> list[dict[str, Any]]:
    import importlib

    rfdetr = importlib.import_module("rfdetr")
    if not hasattr(rfdetr, model_class):
        raise ValueError(f"Unknown RF-DETR model class: {model_class}")
    Model = getattr(rfdetr, model_class)
    model = Model(pretrain_weights=str(weights))
    raw = model.predict(image, threshold=float(conf))
    det = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
    if det is None:
        return []
    xyxy = getattr(det, "xyxy", None)
    if xyxy is None:
        return []
    confs = getattr(det, "confidence", None)
    clss = getattr(det, "class_id", None)
    out: list[dict[str, Any]] = []
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        score = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        cid = int(clss[i]) if clss is not None and i < len(clss) else 0
        out.append(
            {
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": score,
                "category_id": cid if cid in (0, 1) else 0,
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-root", type=str, default="datasets/camponotus_raw")
    p.add_argument(
        "--out",
        type=str,
        default="datasets/camponotus_processed/prelabels/camponotus_prelabels_coco.json",
    )
    p.add_argument("--backend", choices=("auto", "yolo", "rfdetr"), default="auto")
    p.add_argument("--yolo-weights", type=str, default="")
    p.add_argument("--rfdetr-weights", type=str, default="")
    p.add_argument("--rfdetr-model-class", type=str, default="RFDETRSmall")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

    images_root = Path(args.images_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not images_root.is_dir():
        print(f"images root not found: {images_root}", file=sys.stderr)
        sys.exit(1)

    yolo_w = Path(args.yolo_weights).expanduser().resolve() if args.yolo_weights else None
    rfd_w = Path(args.rfdetr_weights).expanduser().resolve() if args.rfdetr_weights else None
    backend = args.backend
    if backend == "auto":
        if rfd_w is not None and rfd_w.is_file():
            backend = "rfdetr"
        elif yolo_w is not None and yolo_w.is_file():
            backend = "yolo"
        else:
            print(
                "auto backend requires at least one valid --yolo-weights or --rfdetr-weights path",
                file=sys.stderr,
            )
            sys.exit(1)

    if backend == "yolo" and (yolo_w is None or not yolo_w.is_file()):
        print("YOLO backend selected but --yolo-weights is missing/invalid", file=sys.stderr)
        sys.exit(1)
    if backend == "rfdetr" and (rfd_w is None or not rfd_w.is_file()):
        print("RF-DETR backend selected but --rfdetr-weights is missing/invalid", file=sys.stderr)
        sys.exit(1)

    images = _collect_images(images_root, args.max_images)
    if not images:
        print("No images found for prelabel generation", file=sys.stderr)
        sys.exit(1)

    coco_images: list[dict[str, Any]] = []
    coco_anns: list[dict[str, Any]] = []
    ann_id = 1
    for i, img_path in enumerate(images, start=1):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if backend == "yolo":
            preds = _infer_with_yolo(yolo_w, rgb, conf=float(args.conf))  # type: ignore[arg-type]
        else:
            preds = _infer_with_rfdetr(
                rfd_w,  # type: ignore[arg-type]
                rgb,
                conf=float(args.conf),
                model_class=str(args.rfdetr_model_class),
            )
        coco_images.append(
            {
                "id": i,
                "file_name": str(img_path.resolve().relative_to(images_root)),
                "width": int(w),
                "height": int(h),
            }
        )
        for pdet in preds:
            coco_anns.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": int(pdet["category_id"]),
                    "bbox": [float(x) for x in pdet["bbox"]],
                    "score": float(pdet["score"]),
                    "area": float(pdet["bbox"][2] * pdet["bbox"][3]),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    payload = {
        "info": {
            "description": "Camponotus prelabels for CVAT correction",
            "backend": backend,
            "classes": CAMPO_CLASSES,
        },
        "images": coco_images,
        "annotations": coco_anns,
        "categories": [{"id": 0, "name": "ant"}, {"id": 1, "name": "trophallaxis"}],
    }
    write_json(out_path, payload)

    manifest = {
        "backend": backend,
        "images_root": str(images_root),
        "output_json": str(out_path),
        "images_processed": len(coco_images),
        "annotations_generated": len(coco_anns),
        "conf": float(args.conf),
        "weights": {
            "yolo": str(yolo_w) if yolo_w else None,
            "rfdetr": str(rfd_w) if rfd_w else None,
        },
    }
    write_json(out_path.with_name(out_path.stem + "_manifest.json"), manifest)
    print(f"Wrote prelabels: {out_path}")
    print(f"Backend: {backend}; images: {len(coco_images)}; annotations: {len(coco_anns)}")


if __name__ == "__main__":
    main()
