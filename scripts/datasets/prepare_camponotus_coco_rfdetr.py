#!/usr/bin/env python3
"""Copy camponotus_yolo split + COCO JSON into Roboflow RF-DETR layout (train/valid + _annotations.coco.json)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(p: Path, root: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def _git_rev(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _copy_one(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _ensure_images_for_split(
    coco: dict[str, Any],
    src_img_dir: Path,
    dst_dir: Path,
    mode: str,
) -> tuple[int, int]:
    """Return (n_expected, n_copied)."""
    n_exp = n_ok = 0
    for im in coco.get("images", []):
        fn = im.get("file_name")
        if not fn:
            continue
        base = Path(str(fn)).name
        src = src_img_dir / base
        dst = dst_dir / base
        n_exp += 1
        if not src.is_file():
            continue
        _copy_one(src, dst, mode)
        n_ok += 1
    return n_exp, n_ok


def _resolve_coco_annotation_files(
    *,
    yolo_root: Path,
    cfg: dict[str, Any],
    root: Path,
) -> tuple[Path, Path]:
    """
    Resolve train/val COCO annotation JSONs with a standardized fallback order:

    1) Explicit `camponotus_coco_annotations_root` from config.
    2) Legacy `yolo_root/annotations/instances_{train,val}.json`.
    3) Paired root convention:
       datasets/camponotus_yolo/<dataset_name>  ->
       datasets/camponotus_coco/<dataset_name>/annotations
    """
    explicit_ann_root = cfg.get("camponotus_coco_annotations_root")
    if explicit_ann_root:
        ann_root = _resolve(Path(str(explicit_ann_root)), root)
        return ann_root / "instances_train.json", ann_root / "instances_val.json"

    legacy_train = yolo_root / "annotations" / "instances_train.json"
    legacy_val = yolo_root / "annotations" / "instances_val.json"
    if legacy_train.is_file() and legacy_val.is_file():
        return legacy_train, legacy_val

    dataset_name = yolo_root.name
    paired_ann_root = root / "datasets" / "camponotus_coco" / dataset_name / "annotations"
    return paired_ann_root / "instances_train.json", paired_ann_root / "instances_val.json"


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=str,
        default=str(root / "configs/datasets/camponotus_coco_rfdetr.yaml"),
        help=(
            "YAML with camponotus_yolo_root, out_root, copy_mode, and optional "
            "camponotus_coco_annotations_root"
        ),
    )
    p.add_argument(
        "--camponotus-yolo-root",
        type=str,
        default=None,
        help="Optional CLI override for camponotus_yolo_root.",
    )
    p.add_argument(
        "--camponotus-coco-annotations-root",
        type=str,
        default=None,
        help=(
            "Optional CLI override for camponotus_coco_annotations_root "
            "(directory containing instances_train.json + instances_val.json)."
        ),
    )
    p.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Optional CLI override for out_root.",
    )
    p.add_argument(
        "--copy-mode",
        type=str,
        choices=("copy", "hardlink"),
        default=None,
        help="Optional CLI override for copy_mode.",
    )
    args = p.parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = dict(raw) if isinstance(raw, dict) else {}

    if args.camponotus_yolo_root is not None:
        cfg["camponotus_yolo_root"] = str(args.camponotus_yolo_root)
    if args.camponotus_coco_annotations_root is not None:
        cfg["camponotus_coco_annotations_root"] = str(args.camponotus_coco_annotations_root)
    if args.out_root is not None:
        cfg["out_root"] = str(args.out_root)
    if args.copy_mode is not None:
        cfg["copy_mode"] = str(args.copy_mode)

    yolo_root = _resolve(Path(str(cfg.get("camponotus_yolo_root", "datasets/camponotus_yolo"))), root)
    out_root = _resolve(Path(str(cfg.get("out_root", "datasets/camponotus_rfdetr_coco"))), root)
    copy_mode = str(cfg.get("copy_mode", "copy")).lower()
    if copy_mode not in ("copy", "hardlink"):
        print("copy_mode must be 'copy' or 'hardlink'", file=sys.stderr)
        sys.exit(1)

    inst_train, inst_val = _resolve_coco_annotation_files(
        yolo_root=yolo_root,
        cfg=cfg,
        root=root,
    )
    img_train = yolo_root / "images" / "train"
    img_val = yolo_root / "images" / "val"
    src_manifest = yolo_root / "prepare_manifest.json"

    for f in (inst_train, inst_val, img_train, img_val):
        if not f.exists():
            print(
                f"Missing required path: {f}\n"
                "Expected either:\n"
                "  - legacy yolo_root/annotations/instances_{train,val}.json, or\n"
                "  - paired root datasets/camponotus_coco/<dataset_name>/annotations, or\n"
                "  - explicit camponotus_coco_annotations_root in config.\n"
                "Run: python3 scripts/datasets/prepare_camponotus_detection_dataset.py ...",
                file=sys.stderr,
            )
            sys.exit(1)

    train_coco: dict[str, Any] = json.loads(inst_train.read_text(encoding="utf-8"))
    val_coco: dict[str, Any] = json.loads(inst_val.read_text(encoding="utf-8"))

    train_dir = out_root / "train"
    valid_dir = out_root / "valid"
    ann_mirror = out_root / "annotations"

    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    ann_mirror.mkdir(parents=True, exist_ok=True)

    te, tc = _ensure_images_for_split(train_coco, img_train, train_dir, copy_mode)
    ve, vc = _ensure_images_for_split(val_coco, img_val, valid_dir, copy_mode)
    if tc != te or vc != ve:
        print(
            f"Warning: train images {tc}/{te}, valid {vc}/{ve} (missing sources under camponotus_yolo?)",
            file=sys.stderr,
        )

    (train_dir / "_annotations.coco.json").write_text(
        json.dumps(train_coco, indent=2), encoding="utf-8"
    )
    (valid_dir / "_annotations.coco.json").write_text(
        json.dumps(val_coco, indent=2), encoding="utf-8"
    )

    shutil.copy2(inst_train, ann_mirror / "instances_train.json")
    shutil.copy2(inst_val, ann_mirror / "instances_val.json")

    man_payload: dict[str, Any] = {
        "camponotus_yolo_root": str(yolo_root),
        "camponotus_coco_annotations_train": str(inst_train),
        "camponotus_coco_annotations_val": str(inst_val),
        "out_root": str(out_root),
        "copy_mode": copy_mode,
        "train_images_expected": te,
        "train_images_copied": tc,
        "val_images_expected": ve,
        "val_images_copied": vc,
        "git_rev": _git_rev(root),
        "source_prepare_manifest": str(src_manifest) if src_manifest.is_file() else None,
    }
    if src_manifest.is_file():
        h = hashlib.sha256(src_manifest.read_bytes()).hexdigest()[:16]
        man_payload["prepare_manifest_sha256_16"] = h

    (out_root / "camponotus_rfdetr_manifest.json").write_text(
        json.dumps(man_payload, indent=2), encoding="utf-8"
    )

    print(f"Wrote RF-DETR COCO export under {out_root}")
    print(f"  train/_annotations.coco.json + {tc} images")
    print(f"  valid/_annotations.coco.json + {vc} images")
    print(f"  annotations/instances_{{train,val}}.json (mirror)")
    print("  camponotus_rfdetr_manifest.json")


if __name__ == "__main__":
    main()
