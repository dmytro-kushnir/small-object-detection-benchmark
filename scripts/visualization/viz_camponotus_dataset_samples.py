#!/usr/bin/env python3
"""Render sample labeled images from Camponotus YOLO dataset splits."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import yaml


CLS = {0: "ant", 1: "trophallaxis"}
COL = {0: (80, 220, 80), 1: (80, 80, 240)}


def _draw_yolo(img, label_file: Path):
    h, w = img.shape[:2]
    if not label_file.is_file():
        return img, []
    classes: list[int] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        xc, yc, nw, nh = [float(v) for v in parts[1:5]]
        bw, bh = nw * w, nh * h
        x1 = int(xc * w - bw / 2)
        y1 = int(yc * h - bh / 2)
        x2 = int(xc * w + bw / 2)
        y2 = int(yc * h + bh / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), COL.get(cid, (220, 220, 220)), 2)
        cv2.putText(
            img,
            CLS.get(cid, str(cid)),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COL.get(cid, (220, 220, 220)),
            1,
            cv2.LINE_AA,
        )
        classes.append(cid)
    return img, classes


def _sample(images: list[Path], n: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    if len(images) <= n:
        return list(images)
    return rng.sample(images, n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-yaml", type=str, default="datasets/camponotus_yolo/dataset.yaml")
    p.add_argument("--out-dir", type=str, default="experiments/visualizations/camponotus_dataset")
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--n-trophallaxis", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dy = Path(args.dataset_yaml).expanduser().resolve()
    if not dy.is_file():
        print(
            f"dataset yaml not found: {dy}. Run Camponotus dataset preparation first.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    cfg = yaml.safe_load(dy.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        print(
            f"dataset yaml is empty or invalid: {dy}. Expected a YAML mapping with key 'path'.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if "path" not in cfg:
        print(
            f"dataset yaml missing required key 'path': {dy}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    root = Path(cfg["path"]).expanduser().resolve()
    train_img_dir = root / "images" / "train"
    val_img_dir = root / "images" / "val"
    if not train_img_dir.is_dir() or not val_img_dir.is_dir():
        print(
            (
                "dataset image directories are missing.\n"
                f"Expected: {train_img_dir} and {val_img_dir}\n"
                "Run dataset preparation first (for example: "
                "./scripts/run_camponotus_dataset_workflow.sh)."
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)

    out_root = Path(args.out_dir).expanduser().resolve()
    out_train = out_root / "train_samples"
    out_val = out_root / "val_samples"
    out_tro = out_root / "trophallaxis_samples"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)
    out_tro.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    train_imgs = sorted(p for p in train_img_dir.iterdir() if p.suffix.lower() in exts)
    val_imgs = sorted(p for p in val_img_dir.iterdir() if p.suffix.lower() in exts)
    tro_candidates: list[Path] = []

    for img_path in _sample(train_imgs, args.n_train, args.seed):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl = root / "labels" / "train" / f"{img_path.stem}.txt"
        painted, classes = _draw_yolo(img, lbl)
        if 1 in classes:
            tro_candidates.append(img_path)
        cv2.imwrite(str(out_train / img_path.name), painted)

    for img_path in _sample(val_imgs, args.n_val, args.seed + 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl = root / "labels" / "val" / f"{img_path.stem}.txt"
        painted, classes = _draw_yolo(img, lbl)
        if 1 in classes:
            tro_candidates.append(img_path)
        cv2.imwrite(str(out_val / img_path.name), painted)

    for img_path in _sample(tro_candidates, args.n_trophallaxis, args.seed + 2):
        split = "train" if (root / "images" / "train" / img_path.name).is_file() else "val"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl = root / "labels" / split / f"{img_path.stem}.txt"
        painted, _ = _draw_yolo(img, lbl)
        cv2.imwrite(str(out_tro / img_path.name), painted)

    print(f"Wrote visualizations under {out_root}")


if __name__ == "__main__":
    main()
