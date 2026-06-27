"""Shared utilities for model-agnostic COCO detection JSON (YOLO, RF-DETR, etc.)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def iter_gt_aligned_image_paths(
    source_dir: Path,
    coco_gt_path: Path,
) -> list[tuple[Path, int]]:
    """Return ``(image_path, image_id)`` for each COCO GT image present under ``source_dir``.

    Iterates COCO ``images[]`` order and joins ``file_name`` basename to ``source_dir``.
    Prefer this over directory walks when YOLO image trees use symlinks whose resolved
    basenames do not match COCO ``file_name`` keys.
    """
    source_dir = source_dir.expanduser().resolve()
    data = json.loads(Path(coco_gt_path).read_text(encoding="utf-8"))
    out: list[tuple[Path, int]] = []
    for im in data.get("images", []):
        fn = im.get("file_name")
        iid = im.get("id")
        if fn is None or iid is None:
            continue
        ip = source_dir / Path(str(fn)).name
        if ip.is_file():
            out.append((ip, int(iid)))
    return out


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
