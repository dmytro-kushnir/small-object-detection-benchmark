#!/usr/bin/env python3
"""Write EXP-A006 summary markdown from compare JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Any, digits: int = 4) -> str:
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x) if x is not None else "—"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare", type=str, default="experiments/results/ants_expA006_vs_baseline.json")
    p.add_argument("--out", type=str, default="experiments/results/ants_expA006_summary.md")
    args = p.parse_args()

    cmp_path = Path(args.compare).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not cmp_path.is_file():
        print(f"Missing compare JSON: {cmp_path}", file=sys.stderr)
        sys.exit(1)
    d = _load(cmp_path)
    pair = d.get("deltas_tracking_minus_baseline", {})
    base = pair.get("baseline_metrics", {})
    comp = pair.get("compare_metrics", {})
    dd = pair.get("deltas", {})
    inf = pair.get("inference_deltas", {})

    lines = [
        "# EXP-A006: RF-DETR + ByteTrack + temporal smoothing (ants val)",
        "",
        "## 1. Method",
        "",
        "- Detector: RF-DETR optimized baseline from EXP-A005.",
        "- Tracking: `supervision.ByteTrack` (sequence-aware resets).",
        "- Smoothing: drop short tracks (<3), fill 1-frame gaps, track-average score.",
        "",
        "## 2. Metrics (absolute)",
        "",
        "| Metric | A005 opt baseline | A006 tracking |",
        "|--------|------------------:|--------------:|",
        f"| mAP@[.5:.95] | {_fmt(base.get('mAP_50_95'))} | {_fmt(comp.get('mAP_50_95'))} |",
        f"| mAP@.5 | {_fmt(base.get('mAP_50'))} | {_fmt(comp.get('mAP_50'))} |",
        f"| mAP_medium | {_fmt(base.get('mAP_medium'))} | {_fmt(comp.get('mAP_medium'))} |",
        f"| Matched P | {_fmt(base.get('precision'))} | {_fmt(comp.get('precision'))} |",
        f"| Matched R | {_fmt(base.get('recall'))} | {_fmt(comp.get('recall'))} |",
        "",
        "## 3. Delta (A006 - baseline)",
        "",
        f"- mAP@[.5:.95]: {_fmt(dd.get('mAP_diff'), 6)}",
        f"- mAP@.5: {_fmt(dd.get('mAP50_diff'), 6)}",
        f"- mAP_medium: {_fmt(dd.get('medium_diff'), 6)}",
        f"- Precision: {_fmt(dd.get('precision_diff'), 6)}",
        f"- Recall: {_fmt(dd.get('recall_diff'), 6)}",
    ]
    if inf.get("fps_diff") is not None:
        lines.append(f"- FPS: {_fmt(inf.get('fps_diff'), 4)}")
    if inf.get("latency_ms_mean_diff") is not None:
        lines.append(f"- Latency mean (ms): {_fmt(inf.get('latency_ms_mean_diff'), 4)}")

    lines.extend(
        [
            "",
            "## 4. Observations",
            "",
            "- Review track overlays and before/after panels for false-positive removal and stability.",
            "- Validate whether recall gains/losses align with filled-gap behavior.",
            "",
            "## 5. Conclusion",
            "",
            "- Decide whether temporal modeling improves detector-only baseline for this dataset/runtime.",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
