#!/usr/bin/env python3
"""Combine per-resolution prediction relative-area stats vs one COCO GT (EXP-A002b)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ants_relative_size_metrics import (
    _load_json,
    compute_ground_truth_block,
    compute_predictions_block,
    _image_sizes,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco-gt", type=str, required=True)
    p.add_argument(
        "--out",
        type=str,
        default="experiments/results/ants_expA002b_relative_metrics.json",
    )
    p.add_argument(
        "--pred",
        dest="preds",
        action="append",
        metavar="IMGSZ=PATH",
        default=[],
        help="Repeatable: 640=path/to/preds.json (imgsz must match sweep)",
    )
    p.add_argument("--score-thr", type=float, default=0.25)
    p.add_argument("--n-bins", type=int, default=40)
    args = p.parse_args()

    gt_path = Path(args.coco_gt).expanduser().resolve()
    if not gt_path.is_file():
        print(f"Missing GT: {gt_path}", file=sys.stderr)
        sys.exit(1)

    pairs: list[tuple[str, Path]] = []
    for item in args.preds:
        if "=" not in item:
            print(f"Expected IMGSZ=PATH, got: {item!r}", file=sys.stderr)
            sys.exit(1)
        k, v = item.split("=", 1)
        k = k.strip()
        pairs.append((k, Path(v).expanduser().resolve()))

    coco = _load_json(gt_path)
    id_to_wh = _image_sizes(coco)
    n_bins = int(args.n_bins)
    thr = float(args.score_thr)

    by_imgsz: dict[str, Any] = {}
    for imgsz_key, pred_path in sorted(pairs, key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        block = compute_predictions_block(pred_path, id_to_wh, n_bins, thr)
        if block is None:
            by_imgsz[imgsz_key] = {"error": f"missing predictions: {pred_path}"}
        else:
            by_imgsz[imgsz_key] = {"predictions": block}

    payload: dict[str, Any] = {
        "experiment": "EXP-A002b",
        "source_gt": str(gt_path),
        "note": (
            "Per-imgsz prediction relative-area distributions (same GT). "
            "Compare means/percentiles across resolutions."
        ),
        "score_threshold": thr,
        "ground_truth": compute_ground_truth_block(coco, n_bins),
        "by_imgsz": by_imgsz,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
