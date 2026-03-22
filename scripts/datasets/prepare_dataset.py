#!/usr/bin/env python3
"""Prepare detection data: COCO JSON + YOLO export, optional small-object filter and resize."""

from __future__ import annotations

import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

try:
    import hydra
    from hydra.utils import get_original_cwd
except ImportError:  # pragma: no cover
    hydra = None  # type: ignore
    get_original_cwd = None  # type: ignore


def load_coco_json(coco_path: Path) -> dict[str, Any]:
    with open(coco_path, encoding="utf-8") as f:
        return json.load(f)


def _gather_images_yolo(yolo_root: Path) -> list[tuple[Path, Path]]:
    """Return list of (image_path, label_path); label may be missing."""
    images_root = yolo_root / "images"
    labels_root = yolo_root / "labels"
    if not images_root.is_dir():
        raise FileNotFoundError(f"YOLO images dir not found: {images_root}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(images_root.rglob("*")):
        if img_path.suffix.lower() not in exts:
            continue
        rel = img_path.relative_to(images_root)
        label_path = labels_root / rel.with_suffix(".txt")
        pairs.append((img_path, label_path))
    return pairs


def gather_images_yolo_split(yolo_root: Path, split: str) -> list[tuple[Path, Path]]:
    """YOLO pairs under images/{split}/ and labels/{split}/ only (for COCO export per split)."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train|val|test, got {split!r}")
    images_sub = yolo_root / "images" / split
    labels_root = yolo_root / "labels" / split
    if not images_sub.is_dir():
        raise FileNotFoundError(f"YOLO images dir not found: {images_sub}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(images_sub.rglob("*")):
        if img_path.suffix.lower() not in exts:
            continue
        rel = img_path.relative_to(images_sub)
        label_path = labels_root / rel.with_suffix(".txt")
        pairs.append((img_path, label_path))
    return pairs


def yolo_to_coco_in_memory_for_split(
    yolo_root: Path,
    data_yaml: Path | None,
    split: str,
) -> dict[str, Any]:
    """Build COCO dict for one split (train / val / test) without mixing image_ids across splits."""
    pairs = gather_images_yolo_split(yolo_root, split)
    names = _load_yolo_names(yolo_root, data_yaml)
    images_out: list[dict[str, Any]] = []
    annotations_out: list[dict[str, Any]] = []
    max_cat = -1
    ann_id = 1
    for img_id, (img_path, label_path) in enumerate(
        tqdm(pairs, desc=f"yolo_to_coco_{split}"), start=1
    ):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        images_out.append(
            {
                "id": img_id,
                "file_name": str(img_path.name),
                "width": int(w),
                "height": int(h),
            }
        )
        if not label_path.is_file():
            continue
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                max_cat = max(max_cat, cls)
                box_w = bw * w
                box_h = bh * h
                x = xc * w - box_w / 2.0
                y = yc * h - box_h / 2.0
                annotations_out.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": [float(x), float(y), float(box_w), float(box_h)],
                        "area": float(box_w * box_h),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    if names:
        categories = [{"id": i, "name": str(names[i])} for i in range(len(names))]
    else:
        categories = [
            {"id": i, "name": f"class_{i}"}
            for i in range(max_cat + 1 if max_cat >= 0 else 1)
        ]

    return {
        "images": images_out,
        "annotations": annotations_out,
        "categories": categories,
    }


def _load_yolo_names(yolo_root: Path, data_yaml: Path | None) -> list[str]:
    candidates = []
    if data_yaml and data_yaml.is_file():
        candidates.append(data_yaml)
    else:
        for name in ("data.yaml", "dataset.yaml"):
            p = yolo_root / name
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return []
    with open(candidates[0], encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names")
    if names is None:
        return []
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return list(names)
    return []


def yolo_to_coco_in_memory(
    yolo_root: Path,
    data_yaml: Path | None,
) -> dict[str, Any]:
    pairs = _gather_images_yolo(yolo_root)
    names = _load_yolo_names(yolo_root, data_yaml)
    images_out: list[dict[str, Any]] = []
    annotations_out: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    max_cat = -1

    ann_id = 1
    for img_id, (img_path, label_path) in enumerate(
        tqdm(pairs, desc="yolo_to_coco"), start=1
    ):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        images_out.append(
            {
                "id": img_id,
                "file_name": str(img_path.name),
                "width": int(w),
                "height": int(h),
                "_abs_path": str(img_path.resolve()),
            }
        )
        if not label_path.is_file():
            continue
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                max_cat = max(max_cat, cls)
                box_w = bw * w
                box_h = bh * h
                x = xc * w - box_w / 2.0
                y = yc * h - box_h / 2.0
                annotations_out.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": [float(x), float(y), float(box_w), float(box_h)],
                        "area": float(box_w * box_h),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    if names:
        categories = [{"id": i, "name": str(names[i])} for i in range(len(names))]
    else:
        categories = [
            {"id": i, "name": f"class_{i}"}
            for i in range(max_cat + 1 if max_cat >= 0 else 1)
        ]

    return {
        "images": images_out,
        "annotations": annotations_out,
        "categories": categories,
    }


def coco_bbox_metrics(
    ann: dict[str, Any], img_w: int, img_h: int
) -> tuple[float, float, float]:
    x, y, bw, bh = ann["bbox"]
    area = float(bw * bh)
    side = math.sqrt(max(area, 0.0))
    img_area = float(max(img_w * img_h, 1))
    frac = area / img_area
    return area, side, frac


def passes_filter(
    ann: dict[str, Any],
    img_w: int,
    img_h: int,
    fcfg: Any,
) -> bool:
    area, side, frac = coco_bbox_metrics(ann, img_w, img_h)
    if fcfg.min_side_px is not None and side < float(fcfg.min_side_px):
        return False
    if fcfg.min_area_px is not None and area < float(fcfg.min_area_px):
        return False
    if fcfg.min_area_frac is not None and frac < float(fcfg.min_area_frac):
        return False
    return True


def filter_annotations(
    coco: dict[str, Any],
    fcfg: Any,
) -> dict[str, Any]:
    id_to_size = {
        im["id"]: (int(im["width"]), int(im["height"])) for im in coco["images"]
    }
    kept = [
        a
        for a in coco["annotations"]
        if passes_filter(a, *id_to_size[a["image_id"]], fcfg)
    ]
    out = dict(coco)
    out["annotations"] = kept
    return out


def split_image_ids(
    image_ids: list[int],
    ratios: Any,
    seed: int,
) -> dict[str, list[int]]:
    r_train, r_val, r_test = float(ratios.train), float(ratios.val), float(ratios.test)
    s = r_train + r_val + r_test
    if s <= 0:
        raise ValueError("split ratios must sum to > 0")
    r_train, r_val, r_test = r_train / s, r_val / s, r_test / s
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * r_train))
    n_val = int(round(n * r_val))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    if r_val > 0 and not val_ids and len(train_ids) > 1:
        val_ids = [train_ids.pop()]
    if r_test > 0 and not test_ids and len(train_ids) > 1:
        test_ids = [train_ids.pop()]
    assert len(train_ids) + len(val_ids) + len(test_ids) == n
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def resolve_image_path(
    coco: dict[str, Any],
    images_dir: Path,
    im: dict[str, Any],
) -> Path:
    if "_abs_path" in im:
        return Path(im["_abs_path"])
    fn = im["file_name"]
    p = Path(fn)
    if p.is_file():
        return p.resolve()
    cand = images_dir / p.name
    if cand.is_file():
        return cand.resolve()
    cand = images_dir / p
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(f"Image not found for {fn} under {images_dir}")


def resize_image_and_boxes(
    img: Any,
    anns: list[dict[str, Any]],
    resize_cfg: Any,
) -> tuple[Any, list[dict[str, Any]], int, int]:
    h, w = img.shape[:2]
    tw = resize_cfg.fixed_width
    th = resize_cfg.fixed_height
    tse = resize_cfg.target_short_edge

    if tw is not None and th is not None:
        nw, nh = int(tw), int(th)
    elif tse is not None:
        tse = int(tse)
        scale = tse / min(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
    else:
        return img, [dict(a) for a in anns], w, h

    sx = nw / w
    sy = nh / h
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    new_anns = []
    for a in anns:
        na = dict(a)
        x, y, bw, bh = na["bbox"]
        na["bbox"] = [x * sx, y * sy, bw * sx, bh * sy]
        na["area"] = float(na["bbox"][2] * na["bbox"][3])
        new_anns.append(na)
    return out, new_anns, nw, nh


def ann_to_yolo_line(yolo_cls: int, ann: dict[str, Any], w: int, h: int) -> str:
    x, y, bw, bh = ann["bbox"]
    xc = (x + bw / 2.0) / max(w, 1)
    yc = (y + bh / 2.0) / max(h, 1)
    nw = bw / max(w, 1)
    nh = bh / max(h, 1)
    return f"{yolo_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def normalize_categories(coco: dict[str, Any]) -> dict[str, Any]:
    """Remap category ids to 0..N-1; update annotations."""
    cats = sorted(coco["categories"], key=lambda c: int(c["id"]))
    old_to_new = {int(c["id"]): i for i, c in enumerate(cats)}
    new_cats = [{"id": i, "name": c["name"]} for i, c in enumerate(cats)]
    new_anns = []
    for a in coco["annotations"]:
        na = dict(a)
        na["category_id"] = old_to_new[int(a["category_id"])]
        new_anns.append(na)
    out = dict(coco)
    out["categories"] = new_cats
    out["annotations"] = new_anns
    return out


def write_dataset_yaml(
    out_path: Path,
    output_dir: Path,
    categories: list[dict[str, Any]],
    splits: dict[str, list[int]],
) -> None:
    names = {c["id"]: c["name"] for c in categories}
    ordered = [names[i] for i in sorted(names)]
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(ordered),
        "names": ordered,
    }
    if splits.get("test"):
        data["test"] = "images/test"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _resolve_path_str(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    base = Path.cwd()
    if get_original_cwd is not None:
        try:
            base = Path(get_original_cwd())
        except Exception:
            pass
    return (base / p).resolve()


def run_prepare(cfg: DictConfig) -> None:
    out_root = _resolve_path_str(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ann_dir = out_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    mode = str(cfg.input.mode).lower()
    if mode == "coco":
        coco_path = _resolve_path_str(cfg.input.coco_json)
        images_dir = _resolve_path_str(cfg.input.images_dir)
        coco = load_coco_json(coco_path)
    elif mode == "yolo":
        yolo_root = _resolve_path_str(cfg.input.yolo_root)
        dy = cfg.input.yolo_data_yaml
        data_yaml = _resolve_path_str(dy) if dy else None
        coco = yolo_to_coco_in_memory(yolo_root, data_yaml)
        images_dir = yolo_root / "images"
    else:
        raise ValueError(f"Unknown input.mode: {mode}")

    apply_to = str(getattr(cfg.filter, "apply_to", "all")).lower()
    train_only_bbox_filter = apply_to in ("train", "train_only")

    if train_only_bbox_filter:
        coco = normalize_categories(coco)
    else:
        coco = filter_annotations(coco, cfg.filter)
        coco = normalize_categories(coco)

    img_to_anns: dict[int, list[dict[str, Any]]] = {}
    for a in coco["annotations"]:
        img_to_anns.setdefault(a["image_id"], []).append(a)

    if cfg.drop_empty_images:
        used_ids = [im["id"] for im in coco["images"] if img_to_anns.get(im["id"])]
    else:
        used_ids = [im["id"] for im in coco["images"]]

    if not used_ids:
        raise ValueError(
            "No images left after filtering; check input paths and filter settings."
        )

    splits = split_image_ids(used_ids, cfg.split, int(cfg.seed))
    split_sets = {k: set(v) for k, v in splits.items()}

    resize_any = (
        cfg.resize.target_short_edge is not None
        or cfg.resize.fixed_width is not None
        or cfg.resize.fixed_height is not None
    )

    split_rows: dict[str, list[dict[str, Any]]] = {k: [] for k in split_sets}

    for split_name, id_set in split_sets.items():
        img_dir = out_root / "images" / split_name
        lbl_dir = out_root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for im in tqdm(
            [x for x in coco["images"] if x["id"] in id_set],
            desc=f"export_{split_name}",
        ):
            src = resolve_image_path(coco, images_dir, im)
            anns = [dict(a) for a in img_to_anns.get(im["id"], [])]
            if resize_any:
                arr = cv2.imread(str(src))
                if arr is None:
                    continue
                arr, anns, im_out_w, im_out_h = resize_image_and_boxes(
                    arr, anns, cfg.resize
                )
                dst = img_dir / src.name
                cv2.imwrite(str(dst), arr)
            elif cfg.copy_images:
                dst = img_dir / src.name
                shutil.copy2(src, dst)
                arr = cv2.imread(str(dst))
                if arr is None:
                    continue
                im_out_h, im_out_w = arr.shape[:2]
            else:
                dst = img_dir / src.name
                if cfg.use_symlink:
                    try:
                        if dst.exists() or dst.is_symlink():
                            dst.unlink()
                        dst.symlink_to(src.resolve())
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
                arr = cv2.imread(str(src))
                if arr is None:
                    continue
                im_out_h, im_out_w = arr.shape[:2]

            if train_only_bbox_filter and split_name == "train":
                anns = [a for a in anns if passes_filter(a, im_out_w, im_out_h, cfg.filter)]

            fname = src.name
            lbl_path = lbl_dir / f"{Path(fname).stem}.txt"
            with open(lbl_path, "w", encoding="utf-8") as lf:
                for a in anns:
                    yc = int(a["category_id"])
                    lf.write(ann_to_yolo_line(yc, a, im_out_w, im_out_h) + "\n")

            split_rows[split_name].append(
                {
                    "file_name": fname,
                    "width": int(im_out_w),
                    "height": int(im_out_h),
                    "annotations": anns,
                }
            )

    for split_name, rows in split_rows.items():
        images_json: list[dict[str, Any]] = []
        annotations_json: list[dict[str, Any]] = []
        ann_id = 1
        for new_im_id, row in enumerate(rows, start=1):
            images_json.append(
                {
                    "id": new_im_id,
                    "file_name": row["file_name"],
                    "width": row["width"],
                    "height": row["height"],
                }
            )
            for a in row["annotations"]:
                x, y, bw, bh = a["bbox"]
                annotations_json.append(
                    {
                        "id": ann_id,
                        "image_id": new_im_id,
                        "category_id": int(a["category_id"]),
                        "bbox": [float(x), float(y), float(bw), float(bh)],
                        "area": float(bw * bh),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        sub = {
            "images": images_json,
            "annotations": annotations_json,
            "categories": list(coco["categories"]),
        }
        out_json = ann_dir / f"instances_{split_name}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(sub, f)

    write_dataset_yaml(
        out_root / "dataset.yaml",
        out_root,
        coco["categories"],
        {k: list(v) for k, v in splits.items()},
    )

    manifest = {
        "seed": int(cfg.seed),
        "input_mode": mode,
        "filter": OmegaConf.to_container(cfg.filter, resolve=True),
        "filter_apply_to": apply_to,
        "train_only_bbox_filter": train_only_bbox_filter,
        "split": OmegaConf.to_container(cfg.split, resolve=True),
        "resize": OmegaConf.to_container(cfg.resize, resolve=True),
        "counts": {s: len(split_rows[s]) for s in split_rows},
        "output_dir": str(out_root),
    }
    with open(out_root / "prepare_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Outputs in {out_root}")


def main() -> None:
    if hydra is None:
        print("hydra-core is required", file=sys.stderr)
        sys.exit(1)


if hydra is not None:

    @hydra.main(
        version_base=None,
        config_path="../../configs",
        config_name="prepare_dataset",
    )
    def _cli(cfg: DictConfig) -> None:
        run_prepare(cfg)


if __name__ == "__main__":
    if hydra is None:
        main()
    else:
        _cli()
