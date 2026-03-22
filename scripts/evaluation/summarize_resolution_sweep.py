#!/usr/bin/env python3
"""Aggregate per-resolution metrics JSONs into sweep summary JSON, recommendation markdown, and plots."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _imgsz_from_path(path: Path) -> int | None:
    m = re.search(r"exp002b_imgsz(\d+)_metrics\.json$", path.name)
    if m:
        return int(m.group(1))
    return None


def _yolo_dir_from_metrics(data: dict[str, Any]) -> str | None:
    w = (data.get("paths") or {}).get("weights")
    if not w:
        return None
    wp = Path(w)
    if len(wp.parts) >= 2 and wp.parts[-2] == "weights":
        return str(wp.parent)
    return str(wp.parent)


def _pred_path_from_metrics(data: dict[str, Any]) -> str | None:
    return (data.get("paths") or {}).get("predictions")


def _summary_row(data: dict[str, Any], imgsz: int) -> dict[str, Any]:
    ce = data.get("coco_eval") or {}
    pr = data.get("matched_pr") or {}
    ib = data.get("inference_benchmark") or {}
    row: dict[str, Any] = {
        "imgsz": imgsz,
        "mAP": ce.get("mAP_50_95"),
        "mAP50": ce.get("mAP_50"),
        "mAP_small": ce.get("mAP_small"),
        "mAP_medium": ce.get("mAP_medium"),
        "mAP_large": ce.get("mAP_large"),
        "precision": pr.get("precision_iou50_score025"),
        "recall": pr.get("recall_iou50_score025"),
        "fps": ib.get("fps"),
        "latency_ms_mean": ib.get("latency_ms_mean"),
    }
    return row


def _collect_inputs(globs: list[str], cwd: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for g in globs:
        for p in sorted(cwd.glob(g)):
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _recommendation_rule(summary: list[dict[str, Any]]) -> tuple[int, str]:
    """Pick imgsz: among rows with fps >= median(fps), highest mAP_small; tie-break lower imgsz."""
    with_fps = [r for r in summary if r.get("fps") is not None]
    if not with_fps:
        best = max(summary, key=lambda r: (r.get("mAP_small") or 0.0, -(r["imgsz"])))
        return best["imgsz"], (
            "No FPS values; fallback: highest mAP_small, then lower imgsz."
        )
    fps_vals = [float(r["fps"]) for r in with_fps]
    med = statistics.median(fps_vals)
    candidates = [r for r in with_fps if float(r["fps"]) >= med]
    candidates.sort(
        key=lambda r: (-(r.get("mAP_small") or 0.0), r["imgsz"]),
    )
    best = candidates[0]
    return best["imgsz"], (
        f"Among resolutions with FPS ≥ median FPS ({med:.4f}), pick highest mAP_small; "
        "tie-break lower imgsz."
    )


def _write_recommendation_md(
    path: Path,
    summary: list[dict[str, Any]],
    chosen_imgsz: int,
    rule_text: str,
) -> None:
    lines = [
        "# EXP-002b: Resolution sweep — recommendation",
        "",
        "Generated from aggregated metrics (same val GT as EXP-000; YOLO26n, 1 epoch per resolution).",
        "",
        "## Summary table (by imgsz)",
        "",
        "| imgsz | mAP | mAP@0.5 | mAP_small | mAP_medium | mAP_large | P | R | FPS | latency ms |",
        "|------:|----:|--------:|----------:|-----------:|----------:|--:|--:|----:|-----------:|",
    ]
    for r in summary:
        def fmt(x: Any) -> str:
            if x is None:
                return "—"
            if isinstance(x, float):
                return f"{x:.4f}" if abs(x) < 100 else f"{x:.2f}"
            return str(x)

        lines.append(
            f"| {r['imgsz']} | {fmt(r.get('mAP'))} | {fmt(r.get('mAP50'))} | "
            f"{fmt(r.get('mAP_small'))} | {fmt(r.get('mAP_medium'))} | {fmt(r.get('mAP_large'))} | "
            f"{fmt(r.get('precision'))} | {fmt(r.get('recall'))} | {fmt(r.get('fps'))} | "
            f"{fmt(r.get('latency_ms_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Best per metric",
            "",
        ]
    )

    def best_key(key: str, maximize: bool = True) -> tuple[int, float | None]:
        rows = [r for r in summary if r.get(key) is not None]
        if not rows:
            return summary[0]["imgsz"], None
        if maximize:
            b = max(rows, key=lambda r: float(r[key]))  # type: ignore[arg-type]
        else:
            b = min(rows, key=lambda r: float(r[key]))  # type: ignore[arg-type]
        return b["imgsz"], float(b[key])  # type: ignore[arg-type]

    ms_i, ms_v = best_key("mAP_small")
    map_i, map_v = best_key("mAP")
    ml_i, ml_v = best_key("mAP_large")
    fps_i, fps_v = best_key("fps")
    lat_i, lat_v = best_key("latency_ms_mean", maximize=False)

    lines.append(f"- **Highest mAP_small:** imgsz **{ms_i}** ({ms_v:.4f})." if ms_v is not None else f"- **Highest mAP_small:** imgsz **{ms_i}**.")
    lines.append(f"- **Highest overall mAP (0.5:0.95):** imgsz **{map_i}** ({map_v:.4f})." if map_v is not None else f"- **Highest overall mAP:** imgsz **{map_i}**.")
    lines.append(f"- **Highest mAP_large:** imgsz **{ml_i}** ({ml_v:.4f})." if ml_v is not None else f"- **Highest mAP_large:** imgsz **{ml_i}**.")
    lines.append(f"- **Fastest FPS:** imgsz **{fps_i}** ({fps_v:.4f})." if fps_v is not None else f"- **Fastest FPS:** imgsz **{fps_i}**.")
    lines.append(f"- **Lowest mean latency:** imgsz **{lat_i}** ({lat_v:.2f} ms)." if lat_v is not None else f"- **Lowest mean latency:** imgsz **{lat_i}**.")

    lines.extend(
        [
            "",
            "## Trade-off recommendation",
            "",
            f"**Suggested imgsz for small-object quality vs speed:** **{chosen_imgsz}**.",
            "",
            f"*Rule used:* {rule_text}",
            "",
            "### How to read this",
            "",
            "- Higher **imgsz** often raises **mAP_small** (finer input) but can hurt **overall mAP** or **mAP_large** after short training, and usually lowers **FPS**.",
            "- Use the table above to see whether a mid resolution improves **mAP_small** without giving up as much **speed** as the largest size.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_sweep(summary: list[dict[str, Any]], plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skip plots.", file=sys.stderr)
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    xs = [r["imgsz"] for r in summary]

    def series(key: str) -> list[float | None]:
        out: list[float | None] = []
        for r in summary:
            v = r.get(key)
            out.append(float(v) if v is not None else None)
        return out

    def plot_one(ylabel: str, key: str, fname: str) -> None:
        ys = series(key)
        y_plot = [y if y is not None else math.nan for y in ys]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, y_plot, marker="o")
        ax.set_xlabel("imgsz")
        ax.set_ylabel(ylabel)
        ax.set_title(f"EXP-002b: {ylabel} vs imgsz")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=120)
        plt.close(fig)

    plot_one("mAP_small", "mAP_small", "exp002b_mapsmall_vs_imgsz.png")
    plot_one("mAP (0.5:0.95)", "mAP", "exp002b_map_vs_imgsz.png")
    plot_one("FPS", "fps", "exp002b_fps_vs_imgsz.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--glob",
        dest="globs",
        action="append",
        default=[],
        help="Glob(s) for metrics JSON (relative to --cwd). Repeatable. Default: exp002b_imgsz*_metrics.json",
    )
    p.add_argument(
        "--cwd",
        type=str,
        default=".",
        help="Working directory for globs",
    )
    p.add_argument(
        "--out",
        type=str,
        default="experiments/results/exp002b_resolution_sweep.json",
        help="Aggregated JSON output path",
    )
    p.add_argument(
        "--recommendation-out",
        type=str,
        default="experiments/results/exp002b_recommendation.md",
        help="Markdown recommendation path",
    )
    p.add_argument(
        "--plots-dir",
        type=str,
        default="experiments/results/plots",
        help="Directory for PNG plots (empty to skip)",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write matplotlib figures",
    )
    args = p.parse_args()

    globs = args.globs if args.globs else ["experiments/results/exp002b_imgsz*_metrics.json"]
    cwd = Path(args.cwd).expanduser().resolve()
    paths = _collect_inputs(globs, cwd)
    if not paths:
        print("No metrics files matched.", file=sys.stderr)
        sys.exit(1)

    runs: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    git_rev: str | None = None
    for path in paths:
        path = path.resolve()
        imgsz = _imgsz_from_path(path)
        if imgsz is None:
            print(f"Skip (cannot parse imgsz from name): {path}", file=sys.stderr)
            continue
        data = _load(path)
        if git_rev is None:
            git_rev = data.get("git_rev")
        runs.append(
            {
                "imgsz": imgsz,
                "metrics_path": str(path),
                "yolo_dir": _yolo_dir_from_metrics(data),
                "pred_path": _pred_path_from_metrics(data),
                "experiment_id": data.get("experiment_id"),
            }
        )
        summary.append(_summary_row(data, imgsz))

    summary.sort(key=lambda r: r["imgsz"])
    runs.sort(key=lambda r: r["imgsz"])

    if not summary:
        print("No valid sweep rows.", file=sys.stderr)
        sys.exit(1)

    chosen_imgsz, rule_text = _recommendation_rule(summary)

    payload: dict[str, Any] = {
        "experiment": "EXP-002b",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": git_rev,
        "note": (
            "Same prepared dataset and val COCO GT as EXP-000 (datasets/processed/test_run). "
            "One YOLO26n train + val infer + evaluate per imgsz."
        ),
        "recommendation": {
            "chosen_imgsz": chosen_imgsz,
            "rule": rule_text,
        },
        "summary": summary,
        "runs": runs,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    rec_path = Path(args.recommendation_out).expanduser().resolve()
    _write_recommendation_md(rec_path, summary, chosen_imgsz, rule_text)
    print(f"Wrote {rec_path}")

    if not args.no_plots and args.plots_dir.strip():
        plot_dir = Path(args.plots_dir).expanduser().resolve()
        _plot_sweep(summary, plot_dir)
        print(f"Wrote plots under {plot_dir}")


if __name__ == "__main__":
    main()
