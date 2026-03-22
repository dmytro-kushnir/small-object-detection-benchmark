#!/usr/bin/env python3
"""Build experiments/results/exp003_sahi_summary.md from compare_metrics JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:+.6f}"
    return str(x)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--compare-baseline",
        type=str,
        required=True,
        help="exp003_sahi_vs_baseline.json",
    )
    p.add_argument(
        "--compare-896",
        type=str,
        required=True,
        help="exp003_sahi_vs_exp002b_896.json",
    )
    p.add_argument(
        "--out",
        type=str,
        default="experiments/results/exp003_sahi_summary.md",
    )
    args = p.parse_args()

    pb = Path(args.compare_baseline).expanduser().resolve()
    p9 = Path(args.compare_896).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    b = _load(pb)
    h = _load(p9)
    db = b.get("deltas") or {}
    dh = h.get("deltas") or {}
    ib = b.get("inference_deltas") or {}
    ih = h.get("inference_deltas") or {}

    lines = [
        "# EXP-003: SAHI sliced inference — summary",
        "",
        "Generated from `compare_metrics.py` outputs. **Compare − baseline** in tables below.",
        "",
        "## vs vanilla EXP-000 (`test_run_metrics.json`)",
        "",
        f"- Baseline experiment: `{b.get('baseline_experiment_id')}`",
        f"- SAHI (test_run weights): `{b.get('compare_experiment_id')}`",
        "",
        "| Metric | Δ (SAHI − vanilla) |",
        "|--------|---------------------|",
        f"| mAP@[.5:.95] | {_fmt(db.get('mAP_diff'))} |",
        f"| mAP@0.5 | {_fmt(db.get('mAP50_diff'))} |",
        f"| mAP_small | {_fmt(db.get('small_diff'))} |",
        f"| mAP_medium | {_fmt(db.get('medium_diff'))} |",
        f"| mAP_large | {_fmt(db.get('large_diff'))} |",
        f"| Precision (matched) | {_fmt(db.get('precision_diff'))} |",
        f"| Recall (matched) | {_fmt(db.get('recall_diff'))} |",
        f"| FPS | {_fmt(ib.get('fps_diff'))} |",
        f"| Latency mean (ms) | {_fmt(ib.get('latency_ms_mean_diff'))} |",
        "",
        f"*Note:* {b.get('evaluation_note', '')}",
        "",
        "## vs vanilla high-res 896 (`exp002b_imgsz896_metrics.json`)",
        "",
        f"- Baseline experiment: `{h.get('baseline_experiment_id')}`",
        f"- SAHI (896-trained weights): `{h.get('compare_experiment_id')}`",
        "",
        "| Metric | Δ (SAHI − vanilla 896) |",
        "|--------|-------------------------|",
        f"| mAP@[.5:.95] | {_fmt(dh.get('mAP_diff'))} |",
        f"| mAP@0.5 | {_fmt(dh.get('mAP50_diff'))} |",
        f"| mAP_small | {_fmt(dh.get('small_diff'))} |",
        f"| mAP_medium | {_fmt(dh.get('medium_diff'))} |",
        f"| mAP_large | {_fmt(dh.get('large_diff'))} |",
        f"| Precision (matched) | {_fmt(dh.get('precision_diff'))} |",
        f"| Recall (matched) | {_fmt(dh.get('recall_diff'))} |",
        f"| FPS | {_fmt(ih.get('fps_diff'))} |",
        f"| Latency mean (ms) | {_fmt(ih.get('latency_ms_mean_diff'))} |",
        "",
        f"*Note:* {h.get('evaluation_note', '')}",
        "",
        "## Interpretation (template)",
        "",
        "- **mAP_small:** Positive deltas suggest SAHI helps small instances on this val set; compare magnitude to the gain from training at higher `imgsz` (EXP-002b vanilla 896 vs test_run vanilla) using your stored metrics JSONs.",
        "- **Overall mAP / mAP_large:** Slicing can add false positives or duplicate boxes; watch precision and large-object AP.",
        "- **Speed:** SAHI usually runs **more** forward passes per image than plain `predict`; expect lower FPS vs vanilla at the same weights unless slices are very large.",
        "- **Vs raising resolution:** High `imgsz` changes the whole-image resize; SAHI keeps full resolution but adds overlap and multiple windows — benefits and costs differ; use FPS and mAP_small together to judge trade-offs for ANTS-style design.",
        "",
        f"**Sources:** [`{pb.name}`]({pb.name}), [`{p9.name}`]({p9.name}).",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
