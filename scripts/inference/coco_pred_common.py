"""Shared utilities for model-agnostic COCO detection JSON (YOLO, RF-DETR, etc.)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_gt_filename_to_image_id(coco_gt_path: Path) -> dict[str, int]:
    """Map COCO ``file_name`` (basename) → ``image_id`` for aligning preds with GT."""
    data = json.loads(coco_gt_path.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for im in data.get("images", []):
        fn = im.get("file_name")
        iid = im.get("id")
        if fn is not None and iid is not None:
            out[Path(str(fn)).name] = int(iid)
    return out


def max_image_id_in_coco(coco_gt_path: Path) -> int:
    """Maximum ``images[].id`` in a COCO JSON, or ``0`` if none."""
    data = json.loads(Path(coco_gt_path).read_text(encoding="utf-8"))
    m = 0
    for im in data.get("images", []):
        iid = im.get("id")
        if iid is not None:
            m = max(m, int(iid))
    return m


def write_coco_predictions_json(path: Path, detections: list[dict[str, Any]]) -> None:
    """Write COCO results list JSON (``evaluate.py`` / ``pycocotools`` compatible)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detections, indent=2), encoding="utf-8")
