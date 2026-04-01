#!/usr/bin/env python3
"""Compare two Idea 2 event evaluation JSON files (A/B)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=str, required=True)
    p.add_argument("--compare", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--evaluation-note", type=str, default="")
    return p.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(compare: float, baseline: float) -> float:
    return float(compare - baseline)


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline).expanduser().resolve()
    compare_path = Path(args.compare).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    b = _load(baseline_path).get("aggregate", {})
    c = _load(compare_path).get("aggregate", {})
    out = {
        "format": "camponotus_idea2_event_compare_v1",
        "baseline": str(baseline_path),
        "compare": str(compare_path),
        "evaluation_note": str(args.evaluation_note),
        "baseline_metrics": b,
        "compare_metrics": c,
        "delta": {
            "precision": _delta(float(c.get("precision", 0.0)), float(b.get("precision", 0.0))),
            "recall": _delta(float(c.get("recall", 0.0)), float(b.get("recall", 0.0))),
            "f1": _delta(float(c.get("f1", 0.0)), float(b.get("f1", 0.0))),
            "mean_tiou_matched": _delta(float(c.get("mean_tiou_matched", 0.0)), float(b.get("mean_tiou_matched", 0.0))),
            "tp": int(c.get("tp", 0)) - int(b.get("tp", 0)),
            "fp": int(c.get("fp", 0)) - int(b.get("fp", 0)),
            "fn": int(c.get("fn", 0)) - int(b.get("fn", 0)),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote Idea 2 event compare bundle: {out_path}")


if __name__ == "__main__":
    main()
