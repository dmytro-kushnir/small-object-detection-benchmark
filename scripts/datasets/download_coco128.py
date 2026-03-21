#!/usr/bin/env python3
"""Download Ultralytics COCO128 (~7MB) into datasets/raw/test/coco128/ (or --output-dir)."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
)
MARKER = ".download_complete"


def download(url: str, dest_zip: Path) -> None:
    req = Request(url, headers={"User-Agent": "small-object-detection-benchmark/1.0"})
    with urlopen(req, timeout=120) as resp:
        dest_zip.write_bytes(resp.read())


def extract_yolo_root(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    # Zip may be coco128/{images,labels,data.yaml} or flat images/
    children = [p for p in out_dir.iterdir() if p.name != MARKER]
    if len(children) == 1 and children[0].is_dir():
        inner = children[0]
        if (inner / "images").is_dir():
            for item in inner.iterdir():
                target = out_dir / item.name
                if target.exists() or target.is_symlink():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            inner.rmdir()


def main() -> None:
    p = argparse.ArgumentParser(description="Download COCO128 YOLO layout for EXP-000 / smoke tests.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/raw/test/coco128"),
        help="YOLO root (will contain images/, labels/, data.yaml)",
    )
    p.add_argument("--url", type=str, default=DEFAULT_URL, help="Zip URL")
    p.add_argument("--force", action="store_true", help="Re-download even if marker exists")
    args = p.parse_args()

    out = args.output_dir.resolve()
    marker = out / MARKER
    if marker.is_file() and not args.force:
        print(f"Already present ({marker}). Use --force to re-download.")
        return

    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "coco128.zip"
        print(f"Downloading {args.url} …")
        download(args.url, zpath)
        # Clear previous extract (keep parent)
        for child in list(out.iterdir()):
            if child.name == MARKER:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        extract_yolo_root(zpath, out)

    marker.write_text("ok\n", encoding="utf-8")
    imgs = out / "images"
    if not imgs.is_dir():
        print(f"Expected {imgs}/ after extract.", file=sys.stderr)
        sys.exit(1)
    n = sum(1 for _ in imgs.rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"Done. YOLO root: {out} ({n} images)")


if __name__ == "__main__":
    main()
