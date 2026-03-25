#!/usr/bin/env python3
"""Rewrite COCO ``images[].file_name`` to paths relative to ``--raw-root`` by scanning ``in_situ/**``.

Use after you place frames under ``datasets/camponotus_raw/in_situ/seq_*/`` so
``prepare_camponotus_detection_dataset.py --split-source manifest`` can match ``splits.json``.

Fails if any basename maps to more than one file (rename on disk to make basenames unique).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", type=str, required=True, help="Input COCO JSON (e.g. CVAT export)")
    p.add_argument(
        "--raw-root",
        type=str,
        default="datasets/camponotus_raw",
        help="Root whose relative paths appear in output file_name (repo: camponotus_raw)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON (default: overwrite --coco if empty)",
    )
    args = p.parse_args()

    coco_path = Path(args.coco).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else coco_path

    if not coco_path.is_file():
        print(f"COCO not found: {coco_path}", file=sys.stderr)
        sys.exit(1)
    if not raw_root.is_dir():
        print(f"raw root not found: {raw_root}", file=sys.stderr)
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    by_base: dict[str, list[str]] = {}
    for path in raw_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.resolve().relative_to(raw_root.resolve()).as_posix()
        by_base.setdefault(path.name, []).append(rel)

    dup = {k: v for k, v in by_base.items() if len(v) > 1}
    if dup:
        print("Duplicate basenames under raw-root (rename files so each basename is unique):", file=sys.stderr)
        for k, v in sorted(dup.items())[:20]:
            print(f"  {k}: {v}", file=sys.stderr)
        if len(dup) > 20:
            print(f"  ... and {len(dup) - 20} more", file=sys.stderr)
        sys.exit(1)

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = coco.get("images")
    if not isinstance(images, list):
        print("COCO has no images list", file=sys.stderr)
        sys.exit(1)

    missing: list[str] = []
    for im in images:
        if not isinstance(im, dict):
            continue
        base = Path(str(im.get("file_name", ""))).name
        rels = by_base.get(base)
        if not rels:
            missing.append(base)
            continue
        im["file_name"] = rels[0]

    if missing:
        print(f"Missing on disk under {raw_root} ({len(missing)} basenames), e.g.: {missing[:10]}", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(images)} images, file_name → relative to {raw_root})")


if __name__ == "__main__":
    main()
