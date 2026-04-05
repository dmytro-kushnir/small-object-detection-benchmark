#!/usr/bin/env python3
"""Verify CVAT COCO + on-disk images before/after align; optional YOLO + RF-DETR export checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _norm_fn(s: str) -> str:
    s = str(s).strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s


def _get_attr(raw: Any, key: str) -> Any:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw.get(key)
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("label") or item.get("key")
            if name is not None and str(name) == key:
                return item.get("value")
    return None


def verify_coco_images(coco_path: Path, raw_root: Path) -> int:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = data.get("images")
    anns = data.get("annotations", [])
    cats = data.get("categories", [])
    if not isinstance(images, list):
        print("error: COCO missing images[]", file=sys.stderr)
        return 1

    missing: list[str] = []
    for im in images:
        if not isinstance(im, dict):
            continue
        fn = _norm_fn(str(im.get("file_name", "")))
        if not fn:
            missing.append("<empty file_name>")
            continue
        p = (raw_root / fn).resolve()
        if not p.is_file():
            missing.append(fn)

    print(f"COCO: {coco_path}")
    print(f"  images: {len(images)}  annotations: {len(anns)}  categories: {len(cats)}")
    if cats:
        for c in cats[:8]:
            if isinstance(c, dict):
                print(f"    category id={c.get('id')} name={c.get('name')!r}")
        if len(cats) > 8:
            print(f"    ... +{len(cats) - 8} more")

    ann_cat = Counter()
    state_vals: Counter[str] = Counter()
    for a in anns:
        if not isinstance(a, dict):
            continue
        try:
            ann_cat[int(a.get("category_id", -1))] += 1
        except (TypeError, ValueError):
            ann_cat[-1] += 1
        st = _get_attr(a.get("attributes"), "state")
        if st is not None:
            state_vals[str(st)] += 1

    if ann_cat:
        top = ", ".join(f"{k}:{v}" for k, v in sorted(ann_cat.items())[:12])
        print(f"  annotation category_id counts: {top}")

    if state_vals:
        print(f"  attributes.state value counts (sample): {dict(state_vals.most_common(8))}")

    if missing:
        u = sorted(set(missing))
        print(
            f"error: {len(missing)} image path(s) missing under raw_root={raw_root}",
            file=sys.stderr,
        )
        for m in u[:20]:
            print(f"  missing: {m}", file=sys.stderr)
        if len(u) > 20:
            print(f"  ... and {len(u) - 20} more", file=sys.stderr)
        return 1

    print(f"ok: all {len(images)} file_name paths resolve under {raw_root}")
    return 0


def verify_yolo_export(yolo_root: Path) -> int:
    dy = yolo_root / "dataset.yaml"
    if not dy.is_file():
        print(f"error: missing {dy}", file=sys.stderr)
        return 1
    if yaml is None:
        print("warn: PyYAML not installed; skipping dataset.yaml nc check", file=sys.stderr)
    else:
        cfg = yaml.safe_load(dy.read_text(encoding="utf-8"))
        nc = cfg.get("nc") if isinstance(cfg, dict) else None
        names = cfg.get("names") if isinstance(cfg, dict) else None
        print(f"YOLO export: {yolo_root}")
        print(f"  dataset.yaml nc={nc} names={names}")
        if nc is not None and int(nc) != 2:
            print(f"error: expected nc=2 for Camponotus normal/troph, got {nc}", file=sys.stderr)
            return 1

    for split in ("train", "val", "test"):
        img_d = yolo_root / "images" / split
        lbl_d = yolo_root / "labels" / split
        if not img_d.is_dir():
            print(f"error: missing {img_d}", file=sys.stderr)
            return 1
        if not lbl_d.is_dir():
            print(f"error: missing {lbl_d}", file=sys.stderr)
            return 1
        n_img = len([p for p in img_d.iterdir() if p.is_file()])
        n_lbl = len([p for p in lbl_d.iterdir() if p.suffix == ".txt"])
        print(f"  {split}: {n_img} images, {n_lbl} label files")
        if n_img == 0:
            print(f"error: empty split {split}", file=sys.stderr)
            return 1

    print("ok: YOLO layout and nc check passed")
    return 0


def verify_rfdetr_export(rfdetr_root: Path) -> int:
    train_ann = rfdetr_root / "train" / "_annotations.coco.json"
    val_ann = rfdetr_root / "valid" / "_annotations.coco.json"
    for p in (train_ann, val_ann):
        if not p.is_file():
            print(f"error: missing {p}", file=sys.stderr)
            return 1
        data = json.loads(p.read_text(encoding="utf-8"))
        cats = data.get("categories", [])
        if not isinstance(cats, list) or len(cats) < 2:
            print(
                f"error: {p} expected >=2 categories for two-class Camponotus, got {len(cats)}",
                file=sys.stderr,
            )
            return 1
        ids = [c.get("id") for c in cats if isinstance(c, dict)]
        print(f"RF-DETR split ann {p.name}: {len(cats)} categories ids={ids}")

    ann_cat_train = Counter()
    for a in json.loads(train_ann.read_text(encoding="utf-8")).get("annotations", []):
        if isinstance(a, dict):
            try:
                ann_cat_train[int(a["category_id"])] += 1
            except (KeyError, TypeError, ValueError):
                pass
    if ann_cat_train:
        print(f"  train annotation category_id counts: {dict(sorted(ann_cat_train.items()))}")

    print(f"ok: RF-DETR Roboflow layout under {rfdetr_root}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coco",
        type=str,
        default=None,
        help="COCO JSON (e.g. aligned export) to check against --raw-root.",
    )
    p.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Bundle root: every images[].file_name must exist as raw-root / file_name.",
    )
    p.add_argument(
        "--prepared-yolo",
        type=str,
        default=None,
        help="After prepare_camponotus_detection_dataset.py: YOLO output root (dataset.yaml).",
    )
    p.add_argument(
        "--prepared-rfdetr",
        type=str,
        default=None,
        help="After prepare_camponotus_coco_rfdetr.py: Roboflow root (train/valid/_annotations).",
    )
    args = p.parse_args()
    code = 0

    if args.coco and args.raw_root:
        coco_path = Path(args.coco).expanduser().resolve()
        raw_root = Path(args.raw_root).expanduser().resolve()
        if not coco_path.is_file():
            print(f"error: --coco not found: {coco_path}", file=sys.stderr)
            return 1
        if not raw_root.is_dir():
            print(f"error: --raw-root not a directory: {raw_root}", file=sys.stderr)
            return 1
        code = verify_coco_images(coco_path, raw_root)
        if code != 0:
            return code

    if args.prepared_yolo:
        yolo_root = Path(args.prepared_yolo).expanduser().resolve()
        code = verify_yolo_export(yolo_root)
        if code != 0:
            return code

    if args.prepared_rfdetr:
        rfdetr_root = Path(args.prepared_rfdetr).expanduser().resolve()
        code = verify_rfdetr_export(rfdetr_root)
        if code != 0:
            return code

    if not args.coco and not args.prepared_yolo and not args.prepared_rfdetr:
        p.print_help()
        print(
            "\nerror: pass at least one of: (--coco and --raw-root), "
            "--prepared-yolo, --prepared-rfdetr",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
