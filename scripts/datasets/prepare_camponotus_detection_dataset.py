#!/usr/bin/env python3
"""Convert corrected Camponotus COCO annotations into YOLO + COCO split exports.

When CVAT uses a single box label (e.g. ``ant``) and encodes trophallaxis via an attribute
(``attributes.state == "trophallaxis"``), exported training classes are still ``0``/``1``:

- ``0`` = ant
- ``1`` = trophallaxis

Optional integer ``track_id`` is copied onto each exported COCO annotation when present (detection
metrics ignore it). Sources: top-level ``track_id``, ``attributes[--track-id-attr]``, or
integer-like ``group_id``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from camponotus_common import (
    CAMPO_CLASSES,
    build_categories,
    normalize_category_id,
    read_json,
    seeded_shuffle,
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


def _resolve_image_src(file_name: str, raw_root: Path) -> Path | None:
    """Return absolute path to image file under raw_root, or None if missing."""
    src = (raw_root / file_name).resolve()
    if src.is_file():
        return src
    candidates = list(raw_root.rglob(Path(file_name).name))
    if not candidates:
        return None
    return candidates[0].resolve()


def _auto_split_labels(n: int, train_ratio: float, val_ratio: float) -> list[str]:
    """Label sequence for shuffled image order; sizes match ``split_camponotus_dataset._split_sequence_names``."""
    if n == 0:
        return []
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1 if n >= 3 else max(0, n - 2)), n)
    n_val = min(max(n_val, 1 if n >= 3 else (1 if n >= 2 else 0)), max(0, n - n_train))
    if n - n_train - n_val == 0 and n >= 3:
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
    n_test = n - n_train - n_val
    return (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)


def _get_attr_value(raw: Any, key: str) -> Any:
    """Read CVAT ``attributes`` as dict or list of ``{name|label|key, value}``."""
    if raw is None or not key:
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


def _parse_int_like(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != int(value):
            return None
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s, 10)
        except ValueError:
            return None
    return None


def _track_id_from_ann(ann: dict[str, Any], attr_name: str) -> int | None:
    tid = _parse_int_like(ann.get("track_id"))
    if tid is not None:
        return tid
    tid = _parse_int_like(_get_attr_value(ann.get("attributes"), attr_name))
    if tid is not None:
        return tid
    return _parse_int_like(ann.get("group_id"))


def _exported_category_id(
    ann: dict[str, Any],
    state_attr: str,
    trophallaxis_state_value: str,
) -> int:
    """If ``state_attr`` is set and the attribute exists on the annotation, map state -> 0/1."""
    if state_attr:
        raw = _get_attr_value(ann.get("attributes"), state_attr)
        if raw is not None:
            s = str(raw).strip().casefold()
            t = str(trophallaxis_state_value).strip().casefold()
            return 1 if s == t else 0
    return int(ann["category_id"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coco-annotations",
        type=str,
        required=True,
        help="Corrected CVAT COCO annotations JSON (all images).",
    )
    p.add_argument("--splits", type=str, default="datasets/camponotus_processed/splits.json")
    p.add_argument(
        "--split-source",
        choices=("manifest", "auto"),
        default="manifest",
        help=(
            "manifest: assign train/val/test using keys in --splits (CVAT file_name must match "
            "seq_*/basename or basename keys). auto: ignore manifest keys; shuffle images that "
            "resolve under --raw-root and split by ratio (for flat CVAT exports)."
        ),
    )
    p.add_argument(
        "--auto-split-seed",
        type=int,
        default=42,
        help="Random seed for --split-source auto (shuffle order before ratio cut).",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="With --split-source auto; default from splits JSON ratios.train or 0.7.",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="With --split-source auto; default from splits JSON ratios.val or 0.15.",
    )
    p.add_argument("--raw-root", type=str, default="datasets/camponotus_raw")
    p.add_argument("--out-yolo", type=str, default="datasets/camponotus_yolo")
    p.add_argument("--out-coco", type=str, default="datasets/camponotus_coco")
    p.add_argument("--copy-mode", choices=("copy", "symlink"), default="symlink")
    p.add_argument("--analysis-out", type=str, default="datasets/camponotus_processed/analysis.json")
    p.add_argument(
        "--state-attr",
        type=str,
        default="state",
        help=(
            "CVAT attribute name for behavior state. When present on an annotation, "
            "exported class is 1 if value matches --trophallaxis-state-value else 0; "
            "when absent, use normalized category_id (legacy two-label CVAT)."
        ),
    )
    p.add_argument(
        "--trophallaxis-state-value",
        type=str,
        default="trophallaxis",
        help="String value of --state-attr that maps to exported class 1 (trophallaxis).",
    )
    p.add_argument(
        "--track-id-attr",
        type=str,
        default="track_id",
        help="When reading CVAT attributes dict/list, use this key for output track_id.",
    )
    p.add_argument(
        "--strip-track-id",
        action="store_true",
        help="Do not write track_id on exported COCO annotations (detection-only JSON).",
    )
    args = p.parse_args()

    coco_path = Path(args.coco_annotations).expanduser().resolve()
    split_path = Path(args.splits).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve()
    out_yolo = Path(args.out_yolo).expanduser().resolve()
    out_coco = Path(args.out_coco).expanduser().resolve()
    if not coco_path.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {coco_path}")
    if str(args.split_source) == "manifest" and not split_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {split_path}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw root not found: {raw_root}")

    coco = read_json(coco_path)
    split_manifest: dict[str, Any] = read_json(split_path) if split_path.is_file() else {}

    img_lookup = _build_image_lookup(coco)
    anns_by_im = _annotations_by_image(coco)
    key_index = _manifest_key_index(split_manifest) if args.split_source == "manifest" else {}

    train_r, val_r = 0.7, 0.15
    ratios = split_manifest.get("ratios", {})
    if isinstance(ratios, dict):
        train_r = float(ratios.get("train", train_r))
        val_r = float(ratios.get("val", val_r))
    if args.train_ratio is not None:
        train_r = float(args.train_ratio)
    if args.val_ratio is not None:
        val_r = float(args.val_ratio)
    if train_r < 0 or val_r < 0 or train_r + val_r > 1.0 + 1e-9:
        raise ValueError("--train-ratio and --val-ratio must be non-negative and sum to at most 1")

    iid_to_src: dict[int, Path] = {}
    for iid, im in sorted(img_lookup.items(), key=lambda kv: kv[0]):
        src = _resolve_image_src(im["file_name"], raw_root)
        if src is not None:
            iid_to_src[iid] = src

    iid_to_split: dict[int, str] = {}
    if args.split_source == "auto":
        ready = [(iid, img_lookup[iid]) for iid in sorted(iid_to_src.keys())]
        shuffled = seeded_shuffle(ready, int(args.auto_split_seed))
        labels = _auto_split_labels(len(shuffled), train_r, val_r)
        for (iid, _), lab in zip(shuffled, labels):
            iid_to_split[iid] = lab

    state_attr = str(args.state_attr).strip()
    troph_val = str(args.trophallaxis_state_value)

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
        if args.split_source == "auto":
            split = iid_to_split.get(iid)
            src = iid_to_src.get(iid)
        else:
            split = _split_for_image(im["file_name"], key_index)
            src = iid_to_src.get(iid)
        if split not in ("train", "val", "test") or src is None:
            continue
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
            cid = _exported_category_id(ann, state_attr, troph_val)
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
            coco_ann: dict[str, Any] = {
                "id": ann_id,
                "image_id": int(iid),
                "category_id": cid,
                "bbox": [float(x) for x in ann["bbox"]],
                "area": float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])),
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
            if not args.strip_track_id:
                tid = _track_id_from_ann(ann, str(args.track_id_attr))
                if tid is not None:
                    coco_ann["track_id"] = int(tid)
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

    tot_ant = sum(counts["class_counts"][s][0] for s in ("train", "val", "test"))
    tot_troph = sum(counts["class_counts"][s][1] for s in ("train", "val", "test"))
    tot_obj = tot_ant + tot_troph
    analysis_payload: dict[str, Any] = {
        "images_by_split": counts["images"],
        "objects_by_split": counts["objects"],
        "class_counts_by_split": counts["class_counts"],
        "source_distribution": counts["source_counts"],
        "class_balance": {
            "ant_fraction": tot_ant / max(1, tot_obj),
            "trophallaxis_fraction": tot_troph / max(1, tot_obj),
        },
        "split_source": str(args.split_source),
    }
    if args.split_source == "auto":
        analysis_payload["auto_split"] = {
            "seed": int(args.auto_split_seed),
            "train_ratio": train_r,
            "val_ratio": val_r,
            "test_ratio": max(0.0, 1.0 - train_r - val_r),
            "coco_images_total": len(img_lookup),
            "resolved_under_raw_root": len(iid_to_src),
        }
    write_json(Path(args.analysis_out).expanduser().resolve(), analysis_payload)

    total_im = sum(counts["images"].values())
    if total_im == 0:
        print(
            "No images were written. Typical causes: (1) COCO file_name does not match "
            "datasets/camponotus_processed/splits.json keys (CVAT often exports flat names "
            "while the manifest uses seq_*/000001.jpg); (2) missing files under --raw-root. "
            "Retry with: --split-source auto",
            file=sys.stderr,
        )

    print(f"Prepared YOLO dataset under: {out_yolo}")
    print(f"Prepared COCO dataset under: {out_coco}")
    print(f"Wrote initial analysis summary: {Path(args.analysis_out).expanduser().resolve()}")


if __name__ == "__main__":
    main()
