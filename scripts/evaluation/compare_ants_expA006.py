#!/usr/bin/env python3
"""EXP-A006 compare: RF-DETR+tracking vs RF-DETR optimized baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from repo_paths import path_for_artifact

EVAL_NOTE = (
    "EXP-A006 compares RF-DETR optimized baseline to RF-DETR+ByteTrack+temporal smoothing "
    "on the same val split and evaluation pipeline."
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pull_eval(m: dict[str, Any]) -> dict[str, float]:
    ce = m.get("coco_eval", {})
    pr = m.get("matched_pr", {})
    return {
        "mAP_50_95": float(ce.get("mAP_50_95", 0.0)),
        "mAP_50": float(ce.get("mAP_50", 0.0)),
        "mAP_medium": float(ce.get("mAP_medium", 0.0)),
        "precision": float(pr.get("precision_iou50_score025", 0.0)),
        "recall": float(pr.get("recall_iou50_score025", 0.0)),
    }


def _pull_infer(m: dict[str, Any]) -> dict[str, float | None]:
    ib = m.get("inference_benchmark", {})
    fps = ib.get("fps")
    lat = ib.get("latency_ms_mean")
    return {
        "fps": float(fps) if fps is not None else None,
        "latency_ms_mean": float(lat) if lat is not None else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--baseline",
        type=str,
        default=str(root / "experiments/results/ants_expA005_optinfer_rfdetr_metrics.json"),
    )
    p.add_argument(
        "--compare",
        type=str,
        default=str(root / "experiments/results/ants_expA006_tracking_metrics.json"),
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(root / "experiments/results/ants_expA006_vs_baseline.json"),
    )
    p.add_argument("--evaluation-note", type=str, default=EVAL_NOTE)
    args = p.parse_args()

    pb = Path(args.baseline).expanduser().resolve()
    pc = Path(args.compare).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not pb.is_file():
        print(f"Baseline metrics missing: {pb}", file=sys.stderr)
        sys.exit(1)
    if not pc.is_file():
        print(f"Compare metrics missing: {pc}", file=sys.stderr)
        sys.exit(1)

    b = _load(pb)
    c = _load(pc)
    vb = _pull_eval(b)
    vc = _pull_eval(c)
    ib = _pull_infer(b)
    ic = _pull_infer(c)

    deltas = {
        "mAP_diff": vc["mAP_50_95"] - vb["mAP_50_95"],
        "mAP50_diff": vc["mAP_50"] - vb["mAP_50"],
        "medium_diff": vc["mAP_medium"] - vb["mAP_medium"],
        "precision_diff": vc["precision"] - vb["precision"],
        "recall_diff": vc["recall"] - vb["recall"],
    }
    inf_d = {
        "fps_diff": (ic["fps"] - ib["fps"]) if ic["fps"] is not None and ib["fps"] is not None else None,
        "latency_ms_mean_diff": (
            ic["latency_ms_mean"] - ib["latency_ms_mean"]
            if ic["latency_ms_mean"] is not None and ib["latency_ms_mean"] is not None
            else None
        ),
    }

    payload: dict[str, Any] = {
        "evaluation_note": args.evaluation_note,
        "paths": {
            "ants_expA005_optinfer_metrics": path_for_artifact(pb, root),
            "ants_expA006_tracking_metrics": path_for_artifact(pc, root),
        },
        "deltas_tracking_minus_baseline": {
            "baseline_experiment_id": b.get("experiment_id"),
            "compare_experiment_id": c.get("experiment_id"),
            "baseline_metrics": vb,
            "compare_metrics": vc,
            "deltas": deltas,
            "baseline_inference": ib,
            "compare_inference": ic,
            "inference_deltas": inf_d,
            "baseline_matched_counts": {
                "tp": b.get("matched_pr", {}).get("tp"),
                "fp": b.get("matched_pr", {}).get("fp"),
                "fn": b.get("matched_pr", {}).get("fn"),
            },
            "compare_matched_counts": {
                "tp": c.get("matched_pr", {}).get("tp"),
                "fp": c.get("matched_pr", {}).get("fp"),
                "fn": c.get("matched_pr", {}).get("fn"),
            },
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
