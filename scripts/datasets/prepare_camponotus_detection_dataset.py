#!/usr/bin/env python3
"""Convert corrected Camponotus COCO annotations into YOLO + COCO split exports."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from camponotus_common import (
    CAMPO_CLASSES,
    build_categories,
    normalize_category_id,
    read_json,
    write_json,
    yolo_line_from_xywh,
)


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _build_image_lookup(coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for im in coco.get("images", []):
        iid = int(im["id"])
        out[iid] = {
            "id": iid,
            "file_name": str(im["file_name"]),
            "width": int(im.get("width", 0)),
            "height": int(im.get("height", 0)),
        }
    return out


def _annotations_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    by_im: dict[int, list[dict[str, Any]]] = defaultdict(list)
    categories = {int(c["id"]): str(c.get("name", "")) for c in coco.get("categories", [])}
    for ann in coco.get("annotations", []):
        a = dict(ann)
        a["category_id"] = normalize_category_id(a.get("category_id"), categories=categories)
        by_im[int(a["image_id"])].append(a)
    return by_im


def _file_key(path_str: str) -> str:
    p = Path(path_str)
    parent = p.parent.name
    return f"{parent}/{p.name}" if parent.startswith("seq_") else p.name


def _manifest_key_index(split_manifest: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    imgs = split_manifest.get("images", {})
    for split in ("train", "val", "test"):
        for p in imgs.get(split, []):
            out[_file_key(str(p))] = split
    return out


def _split_for_image(file_name: str, key_index: dict[str, str]) -> str | None:
    k = _file_key(file_name)
    if k in key_index:
        return key_index[k]
    # fallback to basename only
    b = Path(file_name).name
    return key_index.get(b)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coco-annotations",
        type=str,
        required=True,
        help="Corrected CVAT COCO annotations JSON (all images).",
    )
    p.add_argument("--splits", type=str, default="datasets/camponotus_processed/splits.json")
    p.add_argument("--raw-root", type=str, default="datasets/camponotus_raw")
    p.add_argument("--out-yolo", type=str, default="datasets/camponotus_yolo")
    p.add_argument("--out-coco", type=str, default="datasets/camponotus_coco")
    p.add_argument("--copy-mode", choices=("copy", "symlink"), default="symlink")
    p.add_argument("--analysis-out", type=str, default="datasets/camponotus_processed/analysis.json")
    args = p.parse_args()

    coco_path = Path(args.coco_annotations).expanduser().resolve()
    split_path = Path(args.splits).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve()
    out_yolo = Path(args.out_yolo).expanduser().resolve()
    out_coco = Path(args.out_coco).expanduser().resolve()
    if not coco_path.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {coco_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {split_path}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw root not found: {raw_root}")

    coco = read_json(coco_path)
    split_manifest = read_json(split_path)

    img_lookup = _build_image_lookup(coco)
    anns_by_im = _annotations_by_image(coco)
    key_index = _manifest_key_index(split_manifest)

    # prepare outputs
    for split in ("train", "val", "test"):
        (out_yolo / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_yolo / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_coco / "annotations").mkdir(parents=True, exist_ok=True)

    coco_split_images: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    coco_split_anns: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    ann_id = 1
    counts = {
        "images": {"train": 0, "val": 0, "test": 0},
        "objects": {"train": 0, "val": 0, "test": 0},
        "class_counts": {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}, "test": {0: 0, 1: 0}},
        "source_counts": {"in_situ": 0, "external": 0},
    }

    for iid, im in img_lookup.items():
        split = _split_for_image(im["file_name"], key_index)
        if split not in ("train", "val", "test"):
            continue
        src = (raw_root / im["file_name"]).resolve()
        if not src.is_file():
            # fallback by basename lookup under raw root
            candidates = list(raw_root.rglob(Path(im["file_name"]).name))
            if not candidates:
                continue
            src = candidates[0].resolve()
        dst_name = Path(im["file_name"]).name
        dst_img = out_yolo / "images" / split / dst_name
        _copy_or_link(src, dst_img, mode=str(args.copy_mode))
        counts["images"][split] += 1
        if "/external/" in str(src).replace("\\", "/"):
            counts["source_counts"]["external"] += 1
        else:
            counts["source_counts"]["in_situ"] += 1

        yolo_lines: list[str] = []
        for ann in anns_by_im.get(iid, []):
            cid = int(ann["category_id"])
            line = yolo_line_from_xywh(
                class_id=cid,
                bbox_xywh=[float(x) for x in ann["bbox"]],
                width=int(im["width"]),
                height=int(im["height"]),
            )
            if line is None:
                continue
            yolo_lines.append(line)
            counts["objects"][split] += 1
            counts["class_counts"][split][cid] += 1
            coco_ann = {
                "id": ann_id,
                "image_id": int(iid),
                "category_id": cid,
                "bbox": [float(x) for x in ann["bbox"]],
                "area": float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])),
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
            coco_split_anns[split].append(coco_ann)
            ann_id += 1

        lbl_path = out_yolo / "labels" / split / f"{Path(dst_name).stem}.txt"
        lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        coco_split_images[split].append(
            {
                "id": int(iid),
                "file_name": dst_name,
                "width": int(im["width"]),
                "height": int(im["height"]),
            }
        )

    # YOLO dataset yaml
    yolo_yaml = {
        "path": str(out_yolo),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CAMPO_CLASSES),
        "names": CAMPO_CLASSES,
    }
    (out_yolo / "dataset.yaml").write_text(yaml.dump(yolo_yaml, sort_keys=False), encoding="utf-8")

    for split in ("train", "val", "test"):
        payload = {
            "images": coco_split_images[split],
            "annotations": coco_split_anns[split],
            "categories": build_categories(),
        }
        write_json(out_coco / "annotations" / f"instances_{split}.json", payload)

    analysis_payload = {
        "images_by_split": counts["images"],
        "objects_by_split": counts["objects"],
        "class_counts_by_split": counts["class_counts"],
        "source_distribution": counts["source_counts"],
    }
    write_json(Path(args.analysis_out).expanduser().resolve(), analysis_payload)

    print(f"Prepared YOLO dataset under: {out_yolo}")
    print(f"Prepared COCO dataset under: {out_coco}")
    print(f"Wrote initial analysis summary: {Path(args.analysis_out).expanduser().resolve()}")


if __name__ == "__main__":
    main()
