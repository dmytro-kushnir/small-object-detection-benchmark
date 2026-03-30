#!/usr/bin/env python3
"""Create Camponotus train/val/test split manifest using `track_id` (Idea 2 signal).

This repo’s YOLO training/eval is image-level. Here we implement a heuristic to
use `attributes.track_id` for split assignment anyway:

1. Shuffle/assign track_ids to train/val/test by ratio.
2. For each image, compute the most frequent track_id among its annotations.
3. Assign the entire image to the split of that majority track_id.

This can still cause leakage (a minority track_id inside an image may belong to a different
split). A separate QA script should quantify overlap after splitting.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from camponotus_common import read_json, seeded_shuffle, write_json


def _parse_track_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        # Accept int-like floats/strings produced by exporters.
        return int(value)
    except (TypeError, ValueError):
        return None


def _split_counts(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must be non-negative")
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {s}")

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    # Keep non-empty splits when possible (small-n guardrail).
    if n >= 3 and n_test == 0:
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
        n_test = 1

    if n >= 3 and n_train == 0:
        n_train = 1
        if n_val > 0:
            n_val -= 1
        else:
            n_test = max(0, n_test - 1)

    if n >= 3 and n_val == 0:
        n_val = 1
        if n_train > 0:
            n_train -= 1
        else:
            n_test = max(0, n_test - 1)

    # Clamp (avoid negative due to rounding quirks).
    n_train = max(0, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coco-json",
        type=str,
        required=True,
        help="COCO instances JSON with `annotations[].attributes.track_id`.",
    )
    p.add_argument("--out", type=str, required=True, help="Output splits.json path.")
    p.add_argument("--seed", type=int, default=42, help="Seed for track_id shuffle.")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    args = p.parse_args()

    coco_path = Path(args.coco_json).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    coco: dict[str, Any] = read_json(coco_path)

    images = coco.get("images", [])
    if not isinstance(images, list) or not images:
        raise ValueError("COCO JSON has no images list")

    images_by_id: dict[int, str] = {}
    for im in images:
        iid = int(im["id"])
        images_by_id[iid] = str(im["file_name"])

    track_ids: set[int] = set()
    ann_counts_by_image: dict[int, Counter[int]] = defaultdict(Counter)
    for ann in coco.get("annotations", []):
        iid = int(ann["image_id"])
        attrs = ann.get("attributes") or {}
        tid = None
        if isinstance(attrs, dict):
            tid = _parse_track_id(attrs.get("track_id"))
        if tid is None:
            continue
        track_ids.add(tid)
        ann_counts_by_image[iid][tid] += 1

    if not track_ids:
        raise ValueError("No track_id found in COCO annotations; cannot split by track_id.")

    tracks_sorted = sorted(track_ids)
    tracks_shuffled = seeded_shuffle(tracks_sorted, int(args.seed))
    n_train, n_val, n_test = _split_counts(
        n=len(tracks_shuffled),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )

    train_tracks = set(tracks_shuffled[:n_train])
    val_tracks = set(tracks_shuffled[n_train : n_train + n_val])
    test_tracks = set(tracks_shuffled[n_train + n_val : n_train + n_val + n_test])

    split_by_track: dict[int, str] = {}
    for t in train_tracks:
        split_by_track[t] = "train"
    for t in val_tracks:
        split_by_track[t] = "val"
    for t in test_tracks:
        split_by_track[t] = "test"

    images_by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for iid, file_name in images_by_id.items():
        counts = ann_counts_by_image.get(iid)
        if not counts:
            # Should not happen for this dataset; deterministic fallback.
            images_by_split["train"].append(file_name)
            continue
        # Majority track_id by count; tie-break by smaller track_id for stability.
        majority_tid = min((-c, tid) for tid, c in counts.items())[1]
        split = split_by_track.get(majority_tid)
        images_by_split[split or "train"].append(file_name)

    # Sort file_name for reproducibility.
    for s in ("train", "val", "test"):
        images_by_split[s] = sorted(set(images_by_split[s]))

    payload: dict[str, Any] = {
        "schema_version": 2,
        "policy": {
            "track_assignment": {
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "seed": args.seed,
            },
            "image_assignment": "majority_track_id",
        },
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "images": images_by_split,
        "counts": {s: len(images_by_split[s]) for s in ("train", "val", "test")},
    }

    write_json(out_path, payload)
    print(f"Wrote track_id-majority splits.json: {out_path}")
    print(
        "Images per split:",
        f"train={payload['counts']['train']} val={payload['counts']['val']} test={payload['counts']['test']}",
    )


if __name__ == "__main__":
    main()

