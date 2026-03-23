#!/usr/bin/env python3
"""Shared helpers for Camponotus dataset scripts."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CAMPO_CLASSES = ["ant", "trophallaxis"]
CAMPO_CLASS_TO_ID = {name: idx for idx, name in enumerate(CAMPO_CLASSES)}


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_categories() -> list[dict[str, Any]]:
    return [{"id": i, "name": n, "supercategory": "ant"} for i, n in enumerate(CAMPO_CLASSES)]


def normalize_category_id(cat: Any, categories: dict[int, str] | None = None) -> int:
    if isinstance(cat, str):
        if cat not in CAMPO_CLASS_TO_ID:
            raise ValueError(f"Unknown category name: {cat}")
        return CAMPO_CLASS_TO_ID[cat]
    cid = int(cat)
    if cid in (0, 1):
        return cid
    if categories and cid in categories:
        name = str(categories[cid]).strip().lower()
        if name in CAMPO_CLASS_TO_ID:
            return CAMPO_CLASS_TO_ID[name]
    raise ValueError(f"Unsupported category id: {cid}")


def yolo_line_from_xywh(
    class_id: int,
    bbox_xywh: list[float],
    width: int,
    height: int,
) -> str | None:
    x, y, w, h = map(float, bbox_xywh)
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + max(0.0, w)))
    y2 = max(0.0, min(float(height), y + max(0.0, h)))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.0 or bh <= 0.0:
        return None
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    nw = bw / width
    nh = bh / height
    return f"{int(class_id)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def seeded_shuffle(items: list[Any], seed: int) -> list[Any]:
    out = list(items)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out
