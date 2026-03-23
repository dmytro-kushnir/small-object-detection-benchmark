#!/usr/bin/env python3
"""Compute analysis metrics and basic plots for Camponotus exported dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from camponotus_common import read_json, write_json


def _collect_split_stats(coco: dict[str, Any]) -> dict[str, Any]:
    n_images = len(coco.get("images", []))
    anns = coco.get("annotations", [])
    by_class = {0: 0, 1: 0}
    area_vals: list[float] = []
    objects_per_frame: dict[int, int] = {}
    for ann in anns:
        cid = int(ann.get("category_id", 0))
        if cid not in by_class:
            by_class[cid] = 0
        by_class[cid] += 1
        x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
        area_vals.append(max(0.0, w * h))
        iid = int(ann.get("image_id", -1))
        objects_per_frame[iid] = objects_per_frame.get(iid, 0) + 1
    opf = list(objects_per_frame.values())
    return {
        "images": n_images,
        "objects": len(anns),
        "class_counts": by_class,
        "objects_per_frame_mean": (sum(opf) / len(opf)) if opf else 0.0,
        "objects_per_frame_std": (
            math.sqrt(sum((x - (sum(opf) / len(opf))) ** 2 for x in opf) / len(opf)) if opf else 0.0
        ),
        "bbox_areas": area_vals,
        "trophallaxis_ratio": (by_class.get(1, 0) / max(1, len(anns))),
    }


def _plot_class_counts(class_totals: dict[int, int], out_path: Path) -> None:
    labels = ["ant", "trophallaxis"]
    vals = [int(class_totals.get(0, 0)), int(class_totals.get(1, 0))]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals)
    plt.title("Camponotus Class Counts")
    plt.ylabel("Objects")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_bbox_hist(areas: list[float], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(areas, bins=30)
    plt.title("BBox Area Distribution (px^2)")
    plt.xlabel("Area")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco-root", type=str, default="datasets/camponotus_coco/annotations")
    p.add_argument("--split-manifest", type=str, default="datasets/camponotus_processed/splits.json")
    p.add_argument("--out-json", type=str, default="datasets/camponotus_processed/analysis.json")
    p.add_argument("--plots-dir", type=str, default="datasets/camponotus_processed/plots")
    args = p.parse_args()

    coco_root = Path(args.coco_root).expanduser().resolve()
    split_manifest = Path(args.split_manifest).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    plots_dir = Path(args.plots_dir).expanduser().resolve()
    if not coco_root.is_dir():
        raise FileNotFoundError(f"coco root not found: {coco_root}")
    if not split_manifest.is_file():
        raise FileNotFoundError(f"split manifest not found: {split_manifest}")

    split_data = read_json(split_manifest)
    summary: dict[str, Any] = {
        "splits": {},
        "class_totals": {0: 0, 1: 0},
        "objects_total": 0,
        "images_total": 0,
        "source_distribution": split_data.get("counts", {}),
    }
    all_areas: list[float] = []
    for split in ("train", "val", "test"):
        pjson = coco_root / f"instances_{split}.json"
        if not pjson.is_file():
            continue
        coco = read_json(pjson)
        st = _collect_split_stats(coco)
        summary["splits"][split] = {
            "images": st["images"],
            "objects": st["objects"],
            "class_counts": st["class_counts"],
            "objects_per_frame_mean": st["objects_per_frame_mean"],
            "objects_per_frame_std": st["objects_per_frame_std"],
            "trophallaxis_ratio": st["trophallaxis_ratio"],
        }
        summary["class_totals"][0] += st["class_counts"].get(0, 0)
        summary["class_totals"][1] += st["class_counts"].get(1, 0)
        summary["objects_total"] += st["objects"]
        summary["images_total"] += st["images"]
        all_areas.extend(st["bbox_areas"])

    summary["class_balance"] = {
        "ant_fraction": summary["class_totals"][0] / max(1, summary["objects_total"]),
        "trophallaxis_fraction": summary["class_totals"][1] / max(1, summary["objects_total"]),
    }
    write_json(out_json, summary)

    _plot_class_counts(summary["class_totals"], plots_dir / "class_counts.png")
    _plot_bbox_hist(all_areas, plots_dir / "bbox_area_hist.png")

    print(f"Wrote analysis: {out_json}")
    print(f"Wrote plots: {plots_dir}")


if __name__ == "__main__":
    main()
