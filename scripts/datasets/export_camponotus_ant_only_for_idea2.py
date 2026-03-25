#!/usr/bin/env python3
"""Derive ant-only (single-class) COCO + YOLO exports for Idea 2 training; keep trophallaxis GT on each box.

Reads canonical two-class ``datasets/camponotus_coco`` and ``datasets/camponotus_yolo`` from
``prepare_camponotus_detection_dataset.py``. Writes:

- ``--out-coco-root``: ``annotations/instances_{train,val,test}.json`` with ``categories`` = one class
  ``ant`` (id 0), all ``category_id`` = 0, and per-annotation ``trophallaxis_gt`` (bool) for future
  interaction / event evaluation (non-standard COCO field; pycocotools ignores unknown keys).
- ``--out-yolo-root``: ``dataset.yaml``, ``labels/{split}/*.txt`` with class 0 only; ``images/{split}``
  are symlinks to ``--source-yolo-root/images/{split}`` when ``--link-images symlink``, else copies.

Training Idea 2 detector: use this tree (one class). Evaluating trophallaxis-as-interaction: use
``trophallaxis_gt`` + ``track_id`` (if present) with separate scripts (not yet in repo).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from camponotus_common import read_json, write_json, yolo_line_from_xywh


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


def _ant_only_coco_payload(two_class: dict[str, Any]) -> dict[str, Any]:
    images = two_class.get("images", [])
    categories = [{"id": 0, "name": "ant", "supercategory": "ant"}]
    anns_out: list[dict[str, Any]] = []
    for ann in two_class.get("annotations", []):
        a = dict(ann)
        raw_cid = int(a.get("category_id", 0))
        a["trophallaxis_gt"] = bool(raw_cid == 1)
        a["category_id"] = 0
        anns_out.append(a)
    return {
        "images": images,
        "annotations": anns_out,
        "categories": categories,
        "info": {
            "description": "Camponotus ant-only export for Idea 2 (detector train); trophallaxis_gt on each ann",
            "source": "export_camponotus_ant_only_for_idea2.py",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coco-root",
        type=str,
        default="datasets/camponotus_coco",
        help="Two-class COCO dir with annotations/instances_{train,val,test}.json",
    )
    p.add_argument(
        "--source-yolo-root",
        type=str,
        default="datasets/camponotus_yolo",
        help="Canonical YOLO dataset (images for symlink/copy)",
    )
    p.add_argument(
        "--out-coco-root",
        type=str,
        default="datasets/camponotus_coco_ant_only",
        help="Output directory for ant-only COCO JSON per split",
    )
    p.add_argument(
        "--out-yolo-root",
        type=str,
        default="datasets/camponotus_yolo_ant_only",
        help="Output YOLO layout (labels + optional linked images)",
    )
    p.add_argument(
        "--link-images",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="Populate out-yolo images/ from source-yolo (none = labels + yaml only)",
    )
    args = p.parse_args()

    coco_root = Path(args.coco_root).expanduser().resolve()
    src_yolo = Path(args.source_yolo_root).expanduser().resolve()
    out_coco = Path(args.out_coco_root).expanduser().resolve()
    out_yolo = Path(args.out_yolo_root).expanduser().resolve()

    for split in ("train", "val", "test"):
        inst = coco_root / "annotations" / f"instances_{split}.json"
        if not inst.is_file():
            print(f"Missing {inst}", file=sys.stderr)
            sys.exit(1)
        two = read_json(inst)
        payload = _ant_only_coco_payload(two)
        write_json(out_coco / "annotations" / f"instances_{split}.json", payload)

        src_img_dir = src_yolo / "images" / split
        dst_img_dir = out_yolo / "images" / split
        lbl_dir = out_yolo / "labels" / split
        lbl_dir.mkdir(parents=True, exist_ok=True)

        by_im: dict[int, list[dict[str, Any]]] = {}
        for ann in payload["annotations"]:
            by_im.setdefault(int(ann["image_id"]), []).append(ann)

        im_by_id = {int(im["id"]): im for im in payload["images"]}
        for iid, im in sorted(im_by_id.items(), key=lambda kv: kv[0]):
            fn = Path(str(im["file_name"])).name
            w, h = int(im["width"]), int(im["height"])
            lines: list[str] = []
            for ann in by_im.get(iid, []):
                line = yolo_line_from_xywh(
                    class_id=0,
                    bbox_xywh=[float(x) for x in ann["bbox"]],
                    width=w,
                    height=h,
                )
                if line is not None:
                    lines.append(line)
            (lbl_dir / f"{Path(fn).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            if args.link_images != "none":
                src_f = src_img_dir / fn
                if src_f.is_file():
                    _copy_or_link(src_f, dst_img_dir / fn, mode=str(args.link_images))
                else:
                    print(f"Warning: missing image for ant-only YOLO: {src_f}", file=sys.stderr)

    yolo_yaml = {
        "path": str(out_yolo),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["ant"],
    }
    (out_yolo / "dataset.yaml").write_text(yaml.dump(yolo_yaml, sort_keys=False), encoding="utf-8")

    print(f"Wrote ant-only COCO under {out_coco}/annotations/")
    print(f"Wrote ant-only YOLO under {out_yolo} (images link mode={args.link_images})")


if __name__ == "__main__":
    main()
