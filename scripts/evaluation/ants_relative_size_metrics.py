#!/usr/bin/env python3
"""Relative bbox area metrics: bbox_area / image_area from COCO GT (optional preds)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


# Fraction of image area (heuristic bins for "small on frame" narrative)
REL_BINS: list[tuple[str, float | None, float | None]] = [
    ("very_small", None, 1e-5),
    ("small", 1e-5, 5e-5),
    ("medium", 5e-5, 2e-4),
    ("large", 2e-4, None),
]


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _image_sizes(coco: dict[str, Any]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for im in coco.get("images", []):
        iid = int(im["id"])
        out[iid] = (int(im["width"]), int(im["height"]))
    return out


def _relative_areas_for_anns(
    anns: list[dict[str, Any]],
    id_to_wh: dict[int, tuple[int, int]],
) -> list[float]:
    rels: list[float] = []
    for a in anns:
        iid = int(a["image_id"])
        if iid not in id_to_wh:
            continue
        w, h = id_to_wh[iid]
        area_im = float(max(w * h, 1))
        x, y, bw, bh = a["bbox"]
        area_box = float(max(bw * bh, 0.0))
        rels.append(area_box / area_im)
    return rels


def _histogram(values: list[float], n_bins: int) -> dict[str, Any]:
    if not values:
        return {"bin_edges": [], "counts": [], "n_bins": n_bins}
    mx = max(values)
    hi = min(1.0, mx * 1.001) if mx > 0 else 1.0
    edges = [i * hi / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for v in values:
        if v <= 0:
            continue
        idx = min(int(v / hi * n_bins), n_bins - 1)
        counts[idx] += 1
    return {"bin_edges": edges, "counts": counts, "n_bins": n_bins, "hist_max_rel": hi}


def _percentiles(sorted_vals: list[float], ps: list[float]) -> dict[str, float]:
    if not sorted_vals:
        return {f"p{int(p*100)}": float("nan") for p in ps}
    out: dict[str, float] = {}
    n = len(sorted_vals)
    for p in ps:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            out[f"p{int(p * 100)}"] = sorted_vals[int(k)]
        else:
            out[f"p{int(p * 100)}"] = sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)
    return out


def compute_ground_truth_block(coco: dict[str, Any], n_bins: int) -> dict[str, Any]:
    """Stats block for GT annotations (same shape as payload['ground_truth'])."""
    id_to_wh = _image_sizes(coco)
    gt_anns = list(coco.get("annotations", []))
    gt_rel = _relative_areas_for_anns(gt_anns, id_to_wh)
    gt_sorted = sorted(gt_rel)
    return {
        "n_annotations": len(gt_rel),
        "mean": sum(gt_rel) / len(gt_rel) if gt_rel else None,
        "std": _std(gt_rel),
        "percentiles": _percentiles(gt_sorted, [0.05, 0.5, 0.95]),
        "histogram": _histogram(gt_rel, int(n_bins)),
        "bins_policy": _bin_policy_fractions(gt_rel),
    }


def compute_predictions_block(
    pred_path: Path,
    id_to_wh: dict[int, tuple[int, int]],
    n_bins: int,
    score_thr: float,
) -> dict[str, Any] | None:
    """Predictions stats block; None if file missing."""
    if not pred_path.is_file():
        return None
    raw = _load_json(pred_path)
    preds = raw if isinstance(raw, list) else []
    thr = float(score_thr)
    pred_anns: list[dict[str, Any]] = []
    for pr in preds:
        if float(pr.get("score", 1.0)) < thr:
            continue
        pred_anns.append(
            {
                "image_id": int(pr["image_id"]),
                "bbox": pr["bbox"],
            }
        )
    pr_rel = _relative_areas_for_anns(pred_anns, id_to_wh)
    pr_sorted = sorted(pr_rel)
    return {
        "source": str(pred_path.resolve()),
        "score_threshold": float(score_thr),
        "n_boxes": len(pr_rel),
        "mean": sum(pr_rel) / len(pr_rel) if pr_rel else None,
        "std": _std(pr_rel),
        "percentiles": _percentiles(pr_sorted, [0.05, 0.5, 0.95]),
        "histogram": _histogram(pr_rel, int(n_bins)),
        "bins_policy": _bin_policy_fractions(pr_rel),
    }


def _bin_policy_fractions(values: list[float]) -> list[dict[str, Any]]:
    n = len(values)
    if n == 0:
        return [{"label": lab, "low": lo, "high": hi, "count": 0, "fraction": 0.0} for lab, lo, hi in REL_BINS]
    rows: list[dict[str, Any]] = []
    for lab, lo, hi in REL_BINS:
        cnt = 0
        for v in values:
            if lo is not None and v < lo:
                continue
            if hi is not None and v >= hi:
                continue
            cnt += 1
        rows.append(
            {
                "label": lab,
                "low_exclusive": lo,
                "high_exclusive": hi,
                "count": cnt,
                "fraction": cnt / n,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco-gt", type=str, required=True, help="COCO GT JSON (e.g. instances_val.json)")
    p.add_argument("--out", type=str, default="experiments/results/ants_expA000_relative_metrics.json")
    p.add_argument("--pred", type=str, default=None, help="Optional predictions list JSON for pred relative areas")
    p.add_argument("--score-thr", type=float, default=0.25)
    p.add_argument("--n-bins", type=int, default=40)
    args = p.parse_args()

    gt_path = Path(args.coco_gt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not gt_path.is_file():
        print(f"Missing GT: {gt_path}", file=sys.stderr)
        sys.exit(1)

    coco = _load_json(gt_path)
    id_to_wh = _image_sizes(coco)

    payload: dict[str, Any] = {
        "source_gt": str(gt_path),
        "note": (
            "relative_area = (bbox_w * bbox_h) / (image_w * image_h). "
            "COCOeval mAP_small/mAP_large use fixed pixel-area thresholds on absolute bbox area in px²; "
            "they can be -1 when no GT falls in those bins. This file measures 'small on frame' independently."
        ),
        "ground_truth": compute_ground_truth_block(coco, int(args.n_bins)),
    }

    if args.pred:
        pred_path = Path(args.pred).expanduser().resolve()
        if pred_path.is_file():
            block = compute_predictions_block(
                pred_path, id_to_wh, int(args.n_bins), float(args.score_thr)
            )
            payload["predictions"] = block or {"error": f"missing file: {pred_path}"}
        else:
            payload["predictions"] = {"error": f"missing file: {pred_path}"}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


def _std(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var)


if __name__ == "__main__":
    main()
