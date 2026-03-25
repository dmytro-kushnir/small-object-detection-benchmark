#!/usr/bin/env python3
"""Camponotus: RF-DETR metrics vs YOLO baseline (same compare block as EXP-A005)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVAL_NOTE = (
    "RF-DETR full-image predict vs Ultralytics YOLO on Camponotus two-class data. "
    "Input resolution, augmentation, and backbone may differ; interpret mAP and throughput accordingly."
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
        "baseline_matched_counts": {
            "tp": baseline.get("matched_pr", {}).get("tp"),
            "fp": baseline.get("matched_pr", {}).get("fp"),
            "fn": baseline.get("matched_pr", {}).get("fn"),
        },
        "compare_matched_counts": {
            "tp": compare.get("matched_pr", {}).get("tp"),
            "fp": compare.get("matched_pr", {}).get("fp"),
            "fn": compare.get("matched_pr", {}).get("fn"),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--baseline",
        type=str,
        default=str(root / "experiments/results/camponotus_yolo26n_val_metrics.json"),
        help="YOLO Camponotus metrics JSON",
    )
    p.add_argument(
        "--compare",
        type=str,
        default=str(root / "experiments/results/camponotus_rfdetr_val_metrics.json"),
        help="RF-DETR Camponotus metrics JSON",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(root / "experiments/results/camponotus_rfdetr_val_vs_yolo.json"),
    )
    p.add_argument("--evaluation-note", type=str, default=EVAL_NOTE)
    args = p.parse_args()

    pb = Path(args.baseline).expanduser().resolve()
    pc = Path(args.compare).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not pb.is_file():
        print(f"Baseline metrics not found: {pb}", file=sys.stderr)
        sys.exit(1)
    if not pc.is_file():
        print(f"RF-DETR metrics not found: {pc}", file=sys.stderr)
        sys.exit(1)

    b = _load(pb)
    c = _load(pc)
    pair = _pair_compare(b, c)

    payload: dict[str, Any] = {
        "evaluation_note": args.evaluation_note,
        "paths": {"baseline_metrics": str(pb), "rfdetr_metrics": str(pc)},
        "deltas_rfdetr_minus_yolo": pair,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    d = pair["deltas"]
    print()
    print("=== Camponotus RF-DETR vs YOLO (RF-DETR − YOLO) ===")
    print(f"  mAP@[.5:.95]: {d['mAP_diff']:+.6f}")
    print(f"  mAP@0.5:      {d['mAP50_diff']:+.6f}")
    print(f"  mAP_medium:   {d['medium_diff']:+.6f}")
    inf = pair.get("inference_deltas") or {}
    if inf.get("fps_diff") is not None:
        print(f"  FPS Δ:        {inf['fps_diff']:+.4f}")
    if inf.get("latency_ms_mean_diff") is not None:
        print(f"  Latency Δ ms: {inf['latency_ms_mean_diff']:+.4f}")
    print()


if __name__ == "__main__":
    main()
