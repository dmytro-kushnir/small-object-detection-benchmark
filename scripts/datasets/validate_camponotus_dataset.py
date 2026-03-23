#!/usr/bin/env python3
"""Validate Camponotus YOLO/COCO exports for basic integrity and split sanity."""

from __future__ import annotations

import argparse
from pathlib import Path

from camponotus_common import read_json


def _validate_yolo_labels(labels_dir: Path) -> list[str]:
    errs: list[str] = []
    for lp in sorted(labels_dir.glob("*.txt")):
        for ln_no, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                errs.append(f"{lp}:{ln_no} invalid field count")
                continue
            try:
                cid = int(parts[0])
                vals = [float(x) for x in parts[1:5]]
            except ValueError:
                errs.append(f"{lp}:{ln_no} parse error")
                continue
            if cid not in (0, 1):
                errs.append(f"{lp}:{ln_no} invalid class id {cid}")
            xc, yc, w, h = vals
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                errs.append(f"{lp}:{ln_no} normalized box out of range")
            if w <= 0.0 or h <= 0.0:
                errs.append(f"{lp}:{ln_no} non-positive box size")
    return errs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yolo-root", type=str, default="datasets/camponotus_yolo")
    p.add_argument("--coco-root", type=str, default="datasets/camponotus_coco/annotations")
    p.add_argument("--analysis-json", type=str, default="datasets/camponotus_processed/analysis.json")
    args = p.parse_args()

    yolo_root = Path(args.yolo_root).expanduser().resolve()
    coco_root = Path(args.coco_root).expanduser().resolve()
    analysis_json = Path(args.analysis_json).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    for split in ("train", "val", "test"):
        img_dir = yolo_root / "images" / split
        lbl_dir = yolo_root / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            errors.append(f"Missing YOLO split dirs for {split}")
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.is_file())
        if len(imgs) == 0:
            errors.append(f"Empty split images/{split}")
        for img in imgs:
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                errors.append(f"Missing label file for image: {img}")
        errors.extend(_validate_yolo_labels(lbl_dir))

        coco_path = coco_root / f"instances_{split}.json"
        if not coco_path.is_file():
            errors.append(f"Missing COCO file: {coco_path}")
            continue
        coco = read_json(coco_path)
        if len(coco.get("images", [])) == 0:
            errors.append(f"COCO split has no images: {coco_path}")

    if analysis_json.is_file():
        analysis = read_json(analysis_json)
        cf = float(analysis.get("class_balance", {}).get("trophallaxis_fraction", 0.0))
        if cf < 0.01:
            warnings.append(
                f"Very low trophallaxis fraction ({cf:.4f}); check class balance / label policy."
            )

    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        raise SystemExit(1)
    print("Camponotus dataset validation passed.")


if __name__ == "__main__":
    main()
