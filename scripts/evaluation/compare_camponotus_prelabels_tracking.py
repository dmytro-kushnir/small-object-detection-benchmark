#!/usr/bin/env python3
"""Compare ordinary vs tracked Camponotus prelabels and report tracking quality."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


def _infer_sequence_key(file_name: str) -> str:
    p = Path(file_name)
    for part in reversed(p.parts[:-1]):
        if str(part).startswith("seq_"):
            return str(part)
    stem = p.stem
    m = re.match(r"^(.*?)(\d+)$", stem)
    if m and m.group(1):
        return m.group(1)
    return "__single__"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * q))))
    return float(xs[idx])


def _load_coco(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _per_image_order(images: list[dict[str, Any]]) -> dict[int, tuple[str, int]]:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for im in images:
        iid = int(im["id"])
        fn = str(im["file_name"])
        seq = _infer_sequence_key(fn)
        grouped.setdefault(seq, []).append((iid, fn))
    out: dict[int, tuple[str, int]] = {}
    for seq, items in grouped.items():
        items.sort(key=lambda x: x[1])
        for pos, (iid, _) in enumerate(items):
            out[iid] = (seq, pos)
    return out


def _dataset_stats(coco: dict[str, Any]) -> dict[str, Any]:
    images = list(coco.get("images", []))
    anns = list(coco.get("annotations", []))
    cat_name = {int(c["id"]): str(c["name"]) for c in coco.get("categories", []) if "id" in c and "name" in c}

    by_img: dict[int, list[dict[str, Any]]] = {}
    for a in anns:
        iid = a.get("image_id")
        if iid is None:
            continue
        by_img.setdefault(int(iid), []).append(a)

    per_image_counts = [len(by_img.get(int(im["id"]), [])) for im in images]
    cls_counts: dict[str, int] = {}
    for a in anns:
        cid = int(a.get("category_id", -1))
        name = cat_name.get(cid, str(cid))
        cls_counts[name] = cls_counts.get(name, 0) + 1

    order = _per_image_order(images)
    track_rows = [a for a in anns if a.get("track_id") is not None]
    tracks: dict[tuple[str, int], list[int]] = {}
    for a in track_rows:
        iid = int(a["image_id"])
        seq, frame_pos = order.get(iid, ("__single__", iid))
        tid = int(a["track_id"])
        tracks.setdefault((seq, tid), []).append(frame_pos)

    track_lengths: list[int] = []
    total_gap_events = 0
    total_gap_frames = 0
    for _, positions in tracks.items():
        ps = sorted(positions)
        track_lengths.append(len(ps))
        for i in range(1, len(ps)):
            gap = ps[i] - ps[i - 1] - 1
            if gap > 0:
                total_gap_events += 1
                total_gap_frames += gap

    short_2 = sum(1 for l in track_lengths if l <= 2)
    short_3 = sum(1 for l in track_lengths if l <= 3)
    n_tracks = len(track_lengths)

    return {
        "n_images": len(images),
        "n_annotations": len(anns),
        "images_with_annotations": int(sum(1 for c in per_image_counts if c > 0)),
        "anns_per_image_mean": float(mean(per_image_counts)) if per_image_counts else 0.0,
        "anns_per_image_median": float(median(per_image_counts)) if per_image_counts else 0.0,
        "anns_per_image_p95": _percentile([float(v) for v in per_image_counts], 0.95),
        "category_counts": cls_counts,
        "tracking": {
            "annotations_with_track_id": len(track_rows),
            "track_id_coverage_ratio": (len(track_rows) / len(anns)) if anns else 0.0,
            "unique_tracks": n_tracks,
            "track_len_mean": float(mean(track_lengths)) if track_lengths else 0.0,
            "track_len_median": float(median(track_lengths)) if track_lengths else 0.0,
            "track_len_p90": _percentile([float(v) for v in track_lengths], 0.90),
            "short_tracks_len_le_2": short_2,
            "short_tracks_len_le_3": short_3,
            "short_tracks_ratio_le_2": (short_2 / n_tracks) if n_tracks else 0.0,
            "short_tracks_ratio_le_3": (short_3 / n_tracks) if n_tracks else 0.0,
            "gap_events": total_gap_events,
            "gap_frames_total": total_gap_frames,
        },
    }


def _delta(new: float, old: float) -> float:
    return float(new) - float(old)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ordinary", required=True, help="COCO prelabels without tracking")
    p.add_argument("--tracked", required=True, help="COCO prelabels with --with-tracking")
    p.add_argument("--out-json", required=True, help="Output JSON comparison")
    p.add_argument("--out-txt", default="", help="Optional human-readable text report")
    args = p.parse_args()

    ordinary_path = Path(args.ordinary).expanduser().resolve()
    tracked_path = Path(args.tracked).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve() if args.out_txt else None

    ordinary = _dataset_stats(_load_coco(ordinary_path))
    tracked = _dataset_stats(_load_coco(tracked_path))

    comparison = {
        "ordinary_path": str(ordinary_path),
        "tracked_path": str(tracked_path),
        "ordinary": ordinary,
        "tracked": tracked,
        "delta": {
            "n_annotations": _delta(tracked["n_annotations"], ordinary["n_annotations"]),
            "anns_per_image_mean": _delta(tracked["anns_per_image_mean"], ordinary["anns_per_image_mean"]),
            "images_with_annotations": _delta(
                tracked["images_with_annotations"], ordinary["images_with_annotations"]
            ),
            "track_id_coverage_ratio": _delta(
                tracked["tracking"]["track_id_coverage_ratio"],
                ordinary["tracking"]["track_id_coverage_ratio"],
            ),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    lines = [
        "Camponotus prelabel tracking comparison",
        f"ordinary: {ordinary_path}",
        f"tracked:  {tracked_path}",
        "",
        f"images: {ordinary['n_images']} (ordinary) vs {tracked['n_images']} (tracked)",
        f"annotations: {ordinary['n_annotations']} vs {tracked['n_annotations']} (delta {comparison['delta']['n_annotations']:+.0f})",
        f"ann/image mean: {ordinary['anns_per_image_mean']:.3f} vs {tracked['anns_per_image_mean']:.3f}",
        "",
        "tracked-only continuity indicators:",
        f"- track_id coverage: {tracked['tracking']['track_id_coverage_ratio']:.3%}",
        f"- unique tracks: {tracked['tracking']['unique_tracks']}",
        f"- track len mean/median/p90: {tracked['tracking']['track_len_mean']:.2f} / {tracked['tracking']['track_len_median']:.2f} / {tracked['tracking']['track_len_p90']:.2f}",
        f"- short tracks <=2: {tracked['tracking']['short_tracks_len_le_2']} ({tracked['tracking']['short_tracks_ratio_le_2']:.3%})",
        f"- short tracks <=3: {tracked['tracking']['short_tracks_len_le_3']} ({tracked['tracking']['short_tracks_ratio_le_3']:.3%})",
        f"- gap events: {tracked['tracking']['gap_events']} (total gap frames {tracked['tracking']['gap_frames_total']})",
    ]
    txt_blob = "\n".join(lines) + "\n"
    if out_txt is not None:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(txt_blob, encoding="utf-8")

    print(txt_blob, end="")
    print(f"Wrote JSON: {out_json}")
    if out_txt is not None:
        print(f"Wrote TXT:  {out_txt}")


if __name__ == "__main__":
    main()
