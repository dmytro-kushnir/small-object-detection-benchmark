#!/usr/bin/env python3
"""Load COCO preds → xyxy → merge(empty refined) → COCO; check identity (clamp-only diffs)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_INF_DIR = Path(__file__).resolve().parent
if str(_INF_DIR) not in sys.path:
    sys.path.insert(0, str(_INF_DIR))

from ants_v1.merge import (  # noqa: E402
    coco_list_to_xyxy,
    dets_to_coco_records,
    merge_detections,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else {}


def _norm_rows(recs: list[dict[str, Any]]) -> list[tuple[float, float, float, float, float, int]]:
    rows: list[tuple[float, float, float, float, float, int]] = []
    for d in recs:
        bb = d.get("bbox") or []
        if len(bb) < 4:
            continue
        x, y, w, h = map(float, bb[:4])
        rows.append(
            (
                round(x, 4),
                round(y, 4),
                round(w, 4),
                round(h, 4),
                round(float(d.get("score", 0.0)), 5),
                int(d.get("category_id", 0)),
            )
        )
    rows.sort()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", type=str, required=True)
    p.add_argument("--coco-gt", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/expA004_ants_v1.yaml")
    args = p.parse_args()

    pred_path = Path(args.pred).expanduser().resolve()
    gt_path = Path(args.coco_gt).expanduser().resolve()
    cfg_path = Path(args.config).expanduser().resolve()

    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    coco = json.loads(gt_path.read_text(encoding="utf-8"))
    im_wh: dict[int, tuple[int, int]] = {}
    for im in coco.get("images", []):
        iid = im.get("id")
        if iid is None:
            continue
        im_wh[int(iid)] = (int(im.get("width", 0)), int(im.get("height", 0)))

    by_im: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for d in preds:
        by_im[int(d["image_id"])].append(d)

    cfg = _load_yaml(cfg_path)
    mism = 0
    for im_id, lst in sorted(by_im.items()):
        wh = im_wh.get(im_id, (0, 0))
        w, h = wh
        if w <= 0 or h <= 0:
            print(f"Skip image_id={im_id} (missing WH in GT)")
            continue
        xyxy = coco_list_to_xyxy(lst)
        merged = merge_detections(xyxy, [], [], cfg)
        out = dets_to_coco_records(merged, im_id, w, h)
        a = _norm_rows(lst)
        b = _norm_rows(out)
        if a != b:
            print(f"image_id={im_id}: in {len(a)} out {len(b)}")
            print(f"  sample in  {a[:2]}")
            print(f"  sample out {b[:2]}")
            mism += 1

    print(f"Roundtrip images checked: {len(by_im)}; mismatches={mism}")
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    main()
