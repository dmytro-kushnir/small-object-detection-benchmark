#!/usr/bin/env python3
"""Aggregate ants EXP-A002b per-resolution metrics → sweep JSON, recommendation MD, plots."""

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
    m = re.search(r"ants_expA002b_imgsz(\d+)_metrics\.json$", path.name)
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


def _summary_row_public(data: dict[str, Any], imgsz: int) -> dict[str, Any]:
    """Row shape for ants_expA002b_resolution_sweep.json summary (user-facing keys)."""
    ce = data.get("coco_eval") or {}
    pr = data.get("matched_pr") or {}
    ib = data.get("inference_benchmark") or {}
    lat = ib.get("latency_ms_mean")
    return {
        "imgsz": imgsz,
        "mAP": ce.get("mAP_50_95"),
        "mAP50": ce.get("mAP_50"),
        "mAP_medium": ce.get("mAP_medium"),
        "precision": pr.get("precision_iou50_score025"),
        "recall": pr.get("recall_iou50_score025"),
        "fps": ib.get("fps"),
        "latency_ms": lat,
    }


def _collect_inputs(globs: list[str], cwd: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for g in globs:
        for p in sorted(cwd.glob(g)):
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _map_medium_sort_key(v: Any) -> float:
    """Treat missing / non-finite / COCO -1 as worst for maximization."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("-inf")
    if not math.isfinite(x) or x < 0:
        return float("-inf")
    return x


def _recommendation_tradeoff_mAP_medium(
    summary: list[dict[str, Any]],
) -> tuple[int, str]:
    """Among FPS >= median (finite), pick highest mAP_medium; tie-break lower imgsz."""
    with_fps: list[dict[str, Any]] = []
    for r in summary:
        fps = r.get("fps")
        if fps is None:
            continue
        try:
            fv = float(fps)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            with_fps.append(r)
    if not with_fps:
        best = max(summary, key=lambda r: (_map_medium_sort_key(r.get("mAP_medium")), -r["imgsz"]))
        return best["imgsz"], (
            "No finite FPS values; fallback: highest mAP_medium, then lower imgsz."
        )
    fps_vals = [float(r["fps"]) for r in with_fps]
    med = statistics.median(fps_vals)
    if not math.isfinite(med):
        candidates = list(with_fps)
        rule = (
            "Non-finite median FPS; fallback: among runs with finite FPS, highest mAP_medium; "
            "tie-break lower imgsz."
        )
    else:
        candidates = [r for r in with_fps if float(r["fps"]) >= med]
        if not candidates:
            candidates = list(with_fps)
            rule = (
                "Median-FPS filter yielded no candidates; fallback: among runs with finite FPS, "
                "highest mAP_medium; tie-break lower imgsz."
            )
        else:
            rule = (
                f"Among resolutions with FPS ≥ median FPS ({med:.4f}), pick highest mAP_medium; "
                "tie-break lower imgsz."
            )
    candidates.sort(
        key=lambda r: (-_map_medium_sort_key(r.get("mAP_medium")), r["imgsz"]),
    )
    return candidates[0]["imgsz"], rule


def _write_recommendation_md(
    path: Path,
    summary: list[dict[str, Any]],
    chosen_imgsz: int,
    rule_text: str,
) -> None:
    def fmt(x: Any) -> str:
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.4f}" if abs(x) < 100 else f"{x:.2f}"
        return str(x)

    lines = [
        "# EXP-A002b: Ant resolution sweep — recommendation",
        "",
        "Generated from aggregated metrics (same `datasets/ants_yolo` val GT; YOLO26n, 20 epochs per resolution unless 640 reused from EXP-A000 full).",
        "",
        "## Summary table (by imgsz)",
        "",
        "| imgsz | mAP | mAP@0.5 | mAP_medium | P | R | FPS | latency ms |",
        "|------:|----:|--------:|-----------:|--:|--:|----:|-----------:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['imgsz']} | {fmt(r.get('mAP'))} | {fmt(r.get('mAP50'))} | "
            f"{fmt(r.get('mAP_medium'))} | {fmt(r.get('precision'))} | {fmt(r.get('recall'))} | "
            f"{fmt(r.get('fps'))} | {fmt(r.get('latency_ms'))} |"
        )

    def best_key(key: str, maximize: bool = True) -> tuple[int, Any]:
        rows = [r for r in summary if r.get(key) is not None]
        if not rows:
            return summary[0]["imgsz"], None
        if maximize:
            if key == "mAP_medium":
                b = max(rows, key=lambda r: _map_medium_sort_key(r.get(key)))
            else:
                b = max(rows, key=lambda r: float(r[key]))  # type: ignore[arg-type]
        else:
            b = min(rows, key=lambda r: float(r[key]))  # type: ignore[arg-type]
        return b["imgsz"], b.get(key)

    mm_i, mm_v = best_key("mAP_medium")
    map_i, map_v = best_key("mAP")
    fps_i, fps_v = best_key("fps")
    lat_i, lat_v = best_key("latency_ms", maximize=False)

    lines.extend(
        [
            "",
            "## Best per metric",
            "",
        ]
    )
    lines.append(
        f"- **Highest mAP_medium:** imgsz **{mm_i}** ({fmt(mm_v)})." if mm_v is not None else f"- **Highest mAP_medium:** imgsz **{mm_i}**."
    )
    lines.append(
        f"- **Highest overall mAP (0.5:0.95):** imgsz **{map_i}** ({fmt(map_v)})." if map_v is not None else f"- **Highest overall mAP:** imgsz **{map_i}**."
    )
    lines.append(
        f"- **Fastest FPS:** imgsz **{fps_i}** ({fmt(fps_v)})." if fps_v is not None else f"- **Fastest FPS:** imgsz **{fps_i}**."
    )
    lines.append(
        f"- **Lowest mean latency:** imgsz **{lat_i}** ({fmt(lat_v)} ms)." if lat_v is not None else f"- **Lowest mean latency:** imgsz **{lat_i}**."
    )
    lines.extend(
        [
            "",
            "## Trade-off recommendation",
            "",
            f"**Suggested imgsz (mAP_medium vs median FPS rule):** **{chosen_imgsz}**.",
            "",
            f"*Rule used:* {rule_text}",
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

    def series(key: str) -> list[float]:
        out: list[float] = []
        for r in summary:
            v = r.get(key)
            out.append(float(v) if v is not None and math.isfinite(float(v)) else math.nan)
        return out

    def plot_one(ylabel: str, key: str, fname: str, title_suffix: str) -> None:
        ys = series(key)
        y_plot = [y if not math.isnan(y) else math.nan for y in ys]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, y_plot, marker="o")
        ax.set_xlabel("imgsz")
        ax.set_ylabel(ylabel)
        ax.set_title(f"EXP-A002b: {title_suffix}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=120)
        plt.close(fig)

    plot_one("mAP_medium", "mAP_medium", "ants_expA002b_mapmedium_vs_imgsz.png", "mAP_medium vs imgsz")
    plot_one("mAP (0.5:0.95)", "mAP", "ants_expA002b_map_vs_imgsz.png", "mAP vs imgsz")
    plot_one("FPS", "fps", "ants_expA002b_fps_vs_imgsz.png", "FPS vs imgsz")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--glob",
        dest="globs",
        action="append",
        default=[],
        help="Glob(s) relative to --cwd (default: experiments/results/ants_expA002b_imgsz*_metrics.json)",
    )
    p.add_argument("--cwd", type=str, default=".")
    p.add_argument(
        "--out",
        type=str,
        default="experiments/results/ants_expA002b_resolution_sweep.json",
    )
    p.add_argument(
        "--recommendation-out",
        type=str,
        default="experiments/results/ants_expA002b_recommendation.md",
    )
    p.add_argument(
        "--plots-dir",
        type=str,
        default="experiments/results/plots",
    )
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--baseline-reference",
        type=str,
        default="experiments/results/ants_expA000_full_metrics.json",
    )
    args = p.parse_args()

    globs = args.globs or ["experiments/results/ants_expA002b_imgsz*_metrics.json"]
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
        summary.append(_summary_row_public(data, imgsz))

    summary.sort(key=lambda r: r["imgsz"])
    runs.sort(key=lambda r: r["imgsz"])

    if not summary:
        print("No valid sweep rows.", file=sys.stderr)
        sys.exit(1)

    chosen_imgsz, rule_text = _recommendation_tradeoff_mAP_medium(summary)
    baseline_ref = args.baseline_reference

    payload: dict[str, Any] = {
        "experiment": "EXP-A002b",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": git_rev,
        "baseline_reference": baseline_ref,
        "note": (
            "Same datasets/ants_yolo temporal split and val COCO GT as EXP-A000 full. "
            "One YOLO26n train (20 epochs) + val infer + evaluate per imgsz; 640 may reuse "
            "ants_expA000_full metrics without retraining."
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
