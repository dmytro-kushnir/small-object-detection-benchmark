#!/usr/bin/env python3
"""Draw YOLO GT boxes on a random subset of train/val images (ant dataset)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import yaml


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-yaml",
        type=str,
        default="datasets/ants_yolo/dataset.yaml",
        help="Ultralytics dataset.yaml (path field = dataset root)",
    )
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--n", type=int, default=15, help="Max images to save")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=str,
        default="experiments/visualizations/ants_dataset",
    )
    args = p.parse_args()

    dy = Path(args.dataset_yaml).expanduser().resolve()
    if not dy.is_file():
        print(f"Not found: {dy}", file=sys.stderr)
        sys.exit(1)
    cfg = _load_yaml(dy)
    root = Path(cfg["path"]).expanduser().resolve()
    split = args.split
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    if not img_dir.is_dir():
        print(f"No images: {img_dir}", file=sys.stderr)
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(x for x in img_dir.iterdir() if x.suffix.lower() in exts)
    if not images:
        print("No images found", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    take = min(args.n, len(images))
    chosen = rng.sample(images, take) if len(images) > take else list(images)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in chosen:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        lp = lbl_dir / img_path.with_suffix(".txt").name
        if lp.is_file():
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                xc, yc, nw, nh = map(float, parts[1:5])
                bw, bh = nw * w, nh * h
                x1 = int(xc * w - bw / 2)
                y1 = int(yc * h - bh / 2)
                x2 = int(x1 + bw)
                y2 = int(y1 + bh)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        dst = out_dir / f"gt_{split}_{img_path.name}"
        cv2.imwrite(str(dst), im)

    print(f"Wrote {len(chosen)} images under {out_dir}")


if __name__ == "__main__":
    main()
