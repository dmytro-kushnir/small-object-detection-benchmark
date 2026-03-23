#!/usr/bin/env python3
"""EXP-A004: single JSON with deltas vs vanilla 768 and vs SAHI (optional)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVAL_NOTE = (
    "ANTS v1 times the full pipeline (stage-1 @ base_imgsz + per-ROI refine passes). "
    "Vanilla 768 metrics use a single Ultralytics predict@768. "
    "SAHI metrics use sliced inference per full image when present. "
    "Compare throughput cautiously across these code paths."
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pull_inference(m: dict[str, Any]) -> dict[str, float | None]:
    ib = m.get("inference_benchmark") or {}
    fps = ib.get("fps")
    lat = ib.get("latency_ms_mean")
    return {
        "fps": float(fps) if fps is not None else None,
        "latency_ms_mean": float(lat) if lat is not None else None,
    }


def _infer_deltas(
    vb: dict[str, float | None], vc: dict[str, float | None]
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if vb["fps"] is not None and vc["fps"] is not None:
        out["fps_diff"] = vc["fps"] - vb["fps"]
    else:
        out["fps_diff"] = None
    if vb["latency_ms_mean"] is not None and vc["latency_ms_mean"] is not None:
        out["latency_ms_mean_diff"] = vc["latency_ms_mean"] - vb["latency_ms_mean"]
    else:
        out["latency_ms_mean_diff"] = None
    return out


def _pair_compare(baseline: dict[str, Any], compare: dict[str, Any]) -> dict[str, Any]:
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

    vb = pull(baseline)
    vc = pull(compare)
    deltas = {
        "mAP_diff": vc["mAP_50_95"] - vb["mAP_50_95"],
        "mAP50_diff": vc["mAP_50"] - vb["mAP_50"],
        "small_diff": vc["mAP_small"] - vb["mAP_small"],
        "medium_diff": vc["mAP_medium"] - vb["mAP_medium"],
        "large_diff": vc["mAP_large"] - vb["mAP_large"],
        "precision_diff": vc["precision"] - vb["precision"],
        "recall_diff": vc["recall"] - vb["recall"],
    }
    ib_b = _pull_inference(baseline)
    ib_c = _pull_inference(compare)
    return {
        "baseline_experiment_id": baseline.get("experiment_id"),
        "compare_experiment_id": compare.get("experiment_id"),
        "deltas": deltas,
        "baseline_metrics": vb,
        "compare_metrics": vc,
        "baseline_inference": ib_b,
        "compare_inference": ib_c,
        "inference_deltas": _infer_deltas(ib_b, ib_c),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--metrics-768",
        type=str,
        default=str(root / "experiments/results/ants_expA002b_imgsz768_metrics.json"),
    )
    p.add_argument(
        "--metrics-sahi",
        type=str,
        default=str(root / "experiments/results/ants_expA003_sahi_metrics.json"),
    )
    p.add_argument(
        "--metrics-ants",
        type=str,
        default=str(root / "experiments/results/ants_expA004_ants_metrics.json"),
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(root / "experiments/results/ants_expA004_vs_baseline.json"),
    )
    p.add_argument("--evaluation-note", type=str, default=EVAL_NOTE)
    args = p.parse_args()

    p768 = Path(args.metrics_768).expanduser().resolve()
    psahi = Path(args.metrics_sahi).expanduser().resolve()
    pants = Path(args.metrics_ants).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not p768.is_file():
        print(f"768 metrics not found: {p768}", file=sys.stderr)
        sys.exit(1)
    if not pants.is_file():
        print(f"ANTS metrics not found: {pants}", file=sys.stderr)
        sys.exit(1)

    m768 = _load(p768)
    mants = _load(pants)
    deltas_vs_768 = _pair_compare(m768, mants)

    sahi_ok = psahi.is_file()
    deltas_vs_sahi: dict[str, Any] | None = None
    if sahi_ok:
        msahi = _load(psahi)
        deltas_vs_sahi = _pair_compare(msahi, mants)

    payload: dict[str, Any] = {
        "evaluation_note": args.evaluation_note,
        "paths": {
            "ants_expA002b_imgsz768_metrics": str(p768),
            "ants_expA003_sahi_metrics": str(psahi) if sahi_ok else None,
            "ants_expA004_ants_metrics": str(pants),
        },
        "deltas_vs_768": deltas_vs_768,
        "deltas_vs_sahi": deltas_vs_sahi,
        "sahi_metrics_present": sahi_ok,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    d = deltas_vs_768["deltas"]
    print()
    print("=== EXP-A004 ANTS vs vanilla 768 (ANTS − 768) ===")
    print(f"  mAP@[.5:.95]: {d['mAP_diff']:+.6f}")
    print(f"  mAP@0.5:      {d['mAP50_diff']:+.6f}")
    print(f"  mAP_medium:   {d['medium_diff']:+.6f}")
    if sahi_ok and deltas_vs_sahi is not None:
        d2 = deltas_vs_sahi["deltas"]
        print()
        print("=== EXP-A004 ANTS vs SAHI (ANTS − SAHI) ===")
        print(f"  mAP@[.5:.95]: {d2['mAP_diff']:+.6f}")
        print(f"  mAP@0.5:      {d2['mAP50_diff']:+.6f}")
    else:
        print()
        print("(Skip ANTS vs SAHI: ants_expA003_sahi_metrics.json not found.)")
    print()


if __name__ == "__main__":
    main()
