#!/usr/bin/env python3
"""Quantify `track_id` overlap across splits for Camponotus COCO + splits manifest.

Given:
 - a COCO JSON with `annotations[].attributes.track_id`
 - a `splits.json` manifest with `images.train/val/test`

This reports how many `track_id`s appear in more than one split (leakage proxy),
based on which images are assigned to each split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from camponotus_common import read_json, write_json


def _parse_track_id(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco-json", type=str, required=True, help="COCO instances JSON with track_id in attributes.")
    p.add_argument("--splits-json", type=str, required=True, help="splits.json manifest (must have images{train,val,test}).")
    p.add_argument("--out", type=str, required=True, help="Output JSON path for overlap report.")
    args = p.parse_args()

    coco_path = Path(args.coco_json).expanduser().resolve()
    splits_path = Path(args.splits_json).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    coco: dict[str, Any] = read_json(coco_path)
    splits: dict[str, Any] = read_json(splits_path)

    file_name_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for fn in (splits.get("images", {}).get(split, []) or []):
            file_name_to_split[str(fn)] = split

    images_by_id: dict[int, str] = {}
    for im in coco.get("images", []):
        images_by_id[int(im["id"])] = str(im["file_name"])

    track_to_images: dict[int, set[int]] = defaultdict(set)
    for ann in coco.get("annotations", []):
        attrs = ann.get("attributes") or {}
        tid = _parse_track_id(attrs.get("track_id")) if isinstance(attrs, dict) else None
        if tid is None:
            continue
        track_to_images[tid].add(int(ann["image_id"]))

    track_to_splits: dict[int, set[str]] = defaultdict(set)
    # Count only images that are actually assigned in the split manifest.
    track_to_assigned_image_count: dict[int, int] = defaultdict(int)

    for tid, image_ids in track_to_images.items():
        assigned_ids: set[int] = set()
        for iid in image_ids:
            fn = images_by_id.get(iid)
            if fn is None:
                continue
            sp = file_name_to_split.get(fn)
            if not sp:
                continue
            track_to_splits[tid].add(sp)
            assigned_ids.add(iid)
        track_to_assigned_image_count[tid] = len(assigned_ids)

    overlapping = {tid: s for tid, s in track_to_splits.items() if len(s) > 1}
    non_overlapping = {tid: s for tid, s in track_to_splits.items() if len(s) == 1}

    payload: dict[str, Any] = {
        "coco_json": str(coco_path),
        "splits_json": str(splits_path),
        "track_count_with_any_split": len(track_to_splits),
        "overlapping_track_ids": len(overlapping),
        "non_overlapping_track_ids": len(non_overlapping),
        "overlapping_track_examples": [
            {
                "track_id": tid,
                "splits": sorted(list(s)),
                "image_count": track_to_assigned_image_count.get(tid, 0),
            }
            for tid, s in sorted(
                overlapping.items(),
                key=lambda kv: -track_to_assigned_image_count[kv[0]],
            )[:25]
        ],
    }

    write_json(out_path, payload)
    print(f"Wrote overlap QA report: {out_path}")
    print(f"Overlapping track_ids: {payload['overlapping_track_ids']} / {payload['track_count_with_any_split']}")


if __name__ == "__main__":
    main()
