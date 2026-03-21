#!/usr/bin/env python3
"""Compare two evaluate.py metrics JSONs (EXP-000 vs EXP-001, etc.); write deltas + print summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVAL_NOTE = (
    "EXP-001 applies bbox filtering on the train split only (filter.apply_to: train); "
    "val/test GT match EXP-000 when using the same raw data, seed, and split. "
    "Deltas on val mAP compare the same evaluation labels."
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Diff coco_eval + matched_pr between two metrics JSON files.")
    p.add_argument(
        "--baseline",
        type=str,
        default="experiments/results/test_run_metrics.json",
        help="Baseline metrics JSON (e.g. EXP-000)",
    )
    p.add_argument(
        "--compare",
        type=str,
        default="experiments/results/test_run_exp001_metrics.json",
        help="Experiment metrics JSON (e.g. EXP-001)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="experiments/results/exp001_vs_baseline.json",
        help="Output JSON path",
    )
    args = p.parse_args()

    base_path = Path(args.baseline).expanduser().resolve()
    cmp_path = Path(args.compare).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not base_path.is_file():
        print(f"Baseline not found: {base_path}", file=sys.stderr)
        sys.exit(1)
    if not cmp_path.is_file():
        print(f"Compare file not found: {cmp_path}", file=sys.stderr)
        sys.exit(1)

    b = _load(base_path)
    c = _load(cmp_path)

    def pull(m: dict[str, Any]) -> dict[str, float]:
        ce = m.get("coco_eval", {})
        pr = m.get("matched_pr", {})
        return {
            "mAP_50_95": float(ce.get("mAP_50_95", 0.0)),
            "mAP_50": float(ce.get("mAP_50", 0.0)),
            "mAP_small": float(ce.get("mAP_small", 0.0)),
            "mAP_medium": float(ce.get("mAP_medium", 0.0)),
            "mAP_large": float(ce.get("mAP_large", 0.0)),
            "precision": float(pr.get("precision_iou50_score025", 0.0)),
            "recall": float(pr.get("recall_iou50_score025", 0.0)),
        }

    vb = pull(b)
    vc = pull(c)

    deltas = {
        "mAP_diff": vc["mAP_50_95"] - vb["mAP_50_95"],
        "mAP50_diff": vc["mAP_50"] - vb["mAP_50"],
        "small_diff": vc["mAP_small"] - vb["mAP_small"],
        "medium_diff": vc["mAP_medium"] - vb["mAP_medium"],
        "large_diff": vc["mAP_large"] - vb["mAP_large"],
        "precision_diff": vc["precision"] - vb["precision"],
        "recall_diff": vc["recall"] - vb["recall"],
    }

    payload = {
        "baseline_experiment_id": b.get("experiment_id"),
        "compare_experiment_id": c.get("experiment_id"),
        "deltas": deltas,
        "baseline_metrics": vb,
        "compare_metrics": vc,
        "evaluation_note": EVAL_NOTE,
        "paths": {"baseline": str(base_path), "compare": str(cmp_path)},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    print()
    print("=== EXP-001 vs baseline (compare − baseline) ===")
    print(f"  mAP@[.5:.95]: {deltas['mAP_diff']:+.6f}")
    print(f"  mAP@0.5:      {deltas['mAP50_diff']:+.6f}")
    print(f"  mAP_small:    {deltas['small_diff']:+.6f}  <-- small-object bucket (see note below)")
    print(f"  mAP_medium:   {deltas['medium_diff']:+.6f}")
    print(f"  mAP_large:    {deltas['large_diff']:+.6f}")
    print(f"  P (IoU50):    {deltas['precision_diff']:+.6f}")
    print(f"  R (IoU50):    {deltas['recall_diff']:+.6f}")
    print()
    print("Note:", EVAL_NOTE)
    print()


if __name__ == "__main__":
    main()
