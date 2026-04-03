"""Shared helpers for image discovery and RF-DETR output conversion (infer_yolo / infer_rfdetr)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def iter_image_paths(source: Path) -> list[Path]:
    """List image files under ``source``: single file (if suffix allowed) or non-recursive directory."""
    p = source.expanduser().resolve()
    if p.is_file():
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            return []
        return [p]
    if not p.is_dir():
        return []
    out: list[Path] = []
    for child in p.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
            out.append(child.resolve())
    return sorted(out)


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {k: v for k, v in kwargs.items() if v is not None}
    return {k: v for k, v in kwargs.items() if k in params and v is not None}


def parse_rfdetr_detections(
    det_out: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Parse RF-DETR / supervision ``Detections`` into ``(xyxy, class_id, confidence)``.

    Returns ``None`` if there are no boxes. Arrays are float64 / int64 as appropriate.
    """
    if det_out is None:
        return None
    d = det_out
    if isinstance(d, (list, tuple)) and len(d) > 0:
        d = d[0]

    xyxy = getattr(d, "xyxy", None)
    if xyxy is None:
        return None
    xyxy = np.asarray(xyxy, dtype=np.float64)
    if xyxy.size == 0:
        return None

    n = int(xyxy.shape[0])
    cls = getattr(d, "class_id", None)
    if cls is not None:
        cls = np.asarray(cls, dtype=np.int64).reshape(-1)
    else:
        cls = np.zeros(n, dtype=np.int64)

    conf = getattr(d, "confidence", None)
    if conf is not None:
        conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    else:
        conf = np.ones(n, dtype=np.float64)

    return xyxy, cls, conf


def rfdetr_output_to_coco_records(
    det_out: Any,
    image_id: int,
    category_id_fallback: int = 0,
    class_id_mode: str = "single",
) -> list[dict[str, Any]]:
    """Convert rfdetr/supervision Detections (or similar) to COCO detection dicts."""
    parsed = parse_rfdetr_detections(det_out)
    if parsed is None:
        return []
    xyxy, cls, conf = parsed

    records: list[dict[str, Any]] = []
    n = int(xyxy.shape[0])
    for i in range(n):
        x1, y1, x2, y2 = map(float, xyxy[i].tolist())
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        cid = int(cls[i]) if i < len(cls) else category_id_fallback
        if class_id_mode == "single":
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


def draw_coco_detection_records_on_bgr(
    bgr: np.ndarray,
    records: list[dict[str, Any]],
    *,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    class_names: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw COCO-style detection dicts (``bbox`` xywh, ``category_id``, ``score``) on a BGR image copy."""
    out = bgr.copy()
    h_img, w_img = out.shape[:2]
    palette = (
        (0, 200, 0),
        (0, 0, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
    )

    for rec in records:
        bbox = rec.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, bw, bh = map(float, bbox[:4])
        x2, y2 = x1 + bw, y1 + bh
        x1i = int(max(0, min(w_img - 1, round(x1))))
        y1i = int(max(0, min(h_img - 1, round(y1))))
        x2i = int(max(0, min(w_img - 1, round(x2))))
        y2i = int(max(0, min(h_img - 1, round(y2))))
        if x2i <= x1i or y2i <= y1i:
            continue

        cid = int(rec.get("category_id", 0))
        sc = float(rec.get("score", 1.0))
        color = palette[cid % len(palette)]
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, thickness=line_thickness)

        if class_names and cid in class_names:
            label = f"{class_names[cid]} {sc:.2f}"
        else:
            label = f"{cid} {sc:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(y1i - 4, th + 4)
        cv2.rectangle(out, (x1i, ty - th - 4), (x1i + tw + 4, ty + baseline), color, -1)
        cv2.putText(
            out,
            label,
            (x1i + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def dedupe_paths_preserve_order(paths: list[Path]) -> list[Path]:
    seen: set[Any] = set()
    out: list[Path] = []
    for p in paths:
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out
