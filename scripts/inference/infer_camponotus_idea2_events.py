#!/usr/bin/env python3
"""Infer Camponotus Idea 2 trophallaxis events from MOT JSON tracks."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _bbox_xywh_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox]
    return x, y, x + max(0.0, w), y + max(0.0, h)


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _bbox_xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0


def _center_distance_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    acx, acy = ax + aw * 0.5, ay + ah * 0.5
    bcx, bcy = bx + bw * 0.5, by + bh * 0.5
    return float(math.hypot(acx - bcx, acy - bcy))


def _helper_signal(da: dict[str, Any], db: dict[str, Any], helper_class_id: int) -> float:
    # Idea 1 helper: if either ant in the pair has trophallaxis class signal.
    return 1.0 if int(da.get("category_id", -1)) == helper_class_id or int(db.get("category_id", -1)) == helper_class_id else 0.0


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _merge_runs(runs: list[tuple[int, int]], max_gap_frames: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged: list[tuple[int, int]] = [runs[0]]
    for st, en in runs[1:]:
        pst, pen = merged[-1]
        if st - pen - 1 <= max_gap_frames:
            merged[-1] = (pst, max(pen, en))
        else:
            merged.append((st, en))
    return merged


def _active_runs(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    runs: list[tuple[int, int]] = []
    s = p = int(frames[0])
    for f in frames[1:]:
        f = int(f)
        if f == p + 1:
            p = f
            continue
        runs.append((s, p))
        s = p = f
    runs.append((s, p))
    return runs


def infer_events_for_sequence(
    *,
    sequence_name: str,
    rows: list[dict[str, Any]],
    max_dist_px: float,
    pair_score_threshold: float,
    min_active_frames: int,
    max_gap_frames: int,
    w_iou: float,
    w_dist: float,
    w_helper: float,
    helper_class_id: int,
    use_helper_signal: bool,
) -> list[dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_frame[int(r["frame"])].append(r)

    pair_active_frames: dict[tuple[int, int], list[int]] = defaultdict(list)
    pair_frame_scores: dict[tuple[int, int], list[float]] = defaultdict(list)
    pair_helper_hits: dict[tuple[int, int], int] = defaultdict(int)

    for frame, dets in by_frame.items():
        dets_sorted = sorted(dets, key=lambda x: int(x["track_id"]))
        n = len(dets_sorted)
        for i in range(n):
            for j in range(i + 1, n):
                da = dets_sorted[i]
                db = dets_sorted[j]
                ta = int(da["track_id"])
                tb = int(db["track_id"])
                key = _pair_key(ta, tb)
                iou = _iou_xywh(da["bbox"], db["bbox"])
                dist = _center_distance_xywh(da["bbox"], db["bbox"])
                dist_term = 1.0 - min(1.0, float(dist / max_dist_px)) if max_dist_px > 0 else 0.0
                helper = _helper_signal(da, db, helper_class_id) if use_helper_signal else 0.0
                score = float(w_iou * iou + w_dist * dist_term + w_helper * helper)
                if score >= pair_score_threshold:
                    pair_active_frames[key].append(frame)
                    pair_frame_scores[key].append(score)
                    if helper > 0.0:
                        pair_helper_hits[key] += 1

    events: list[dict[str, Any]] = []
    for key, frames in pair_active_frames.items():
        runs = _active_runs(sorted(frames))
        runs = _merge_runs(runs, max_gap_frames=max_gap_frames)
        scores = pair_frame_scores[key]
        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        max_score = float(max(scores)) if scores else 0.0
        for idx, (st, en) in enumerate(runs, start=1):
            dur = int(en - st + 1)
            if dur < min_active_frames:
                continue
            events.append(
                {
                    "event_id": f"{sequence_name}_{key[0]}_{key[1]}_{idx}",
                    "sequence_name": sequence_name,
                    "track_id_a": int(key[0]),
                    "track_id_b": int(key[1]),
                    "start_frame": int(st),
                    "end_frame": int(en),
                    "duration_frames": int(dur),
                    "confidence_mean": mean_score,
                    "confidence_max": max_score,
                    "helper_support_frames": int(pair_helper_hits.get(key, 0)),
                }
            )
    return sorted(events, key=lambda x: (x["sequence_name"], x["start_frame"], x["track_id_a"], x["track_id_b"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mot-json", type=str, required=True, help="MOT JSON produced by camponotus tracking export.")
    p.add_argument("--out", type=str, required=True, help="Output JSON path for inferred events.")
    p.add_argument("--max-dist-px", type=float, default=90.0)
    p.add_argument("--pair-score-threshold", type=float, default=0.45)
    p.add_argument("--min-active-frames", type=int, default=12)
    p.add_argument("--max-gap-frames", type=int, default=3)
    p.add_argument("--w-iou", type=float, default=0.55)
    p.add_argument("--w-dist", type=float, default=0.35)
    p.add_argument("--w-helper", type=float, default=0.10)
    p.add_argument("--helper-class-id", type=int, default=1, help="Class id used as trophallaxis helper signal.")
    p.add_argument("--disable-helper-signal", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mot_path = Path(args.mot_json).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    payload = json.loads(mot_path.read_text(encoding="utf-8"))
    seqs = payload.get("sequences", [])

    all_events: list[dict[str, Any]] = []
    per_sequence: dict[str, int] = {}
    for seq in seqs:
        sequence_name = str(seq.get("sequence_name", "unknown"))
        rows = list(seq.get("rows", []))
        events = infer_events_for_sequence(
            sequence_name=sequence_name,
            rows=rows,
            max_dist_px=float(args.max_dist_px),
            pair_score_threshold=float(args.pair_score_threshold),
            min_active_frames=int(args.min_active_frames),
            max_gap_frames=int(args.max_gap_frames),
            w_iou=float(args.w_iou),
            w_dist=float(args.w_dist),
            w_helper=float(args.w_helper),
            helper_class_id=int(args.helper_class_id),
            use_helper_signal=not bool(args.disable_helper_signal),
        )
        all_events.extend(events)
        per_sequence[sequence_name] = len(events)

    out = {
        "format": "camponotus_idea2_events_v1",
        "source_mot_json": str(mot_path),
        "config": {
            "max_dist_px": float(args.max_dist_px),
            "pair_score_threshold": float(args.pair_score_threshold),
            "min_active_frames": int(args.min_active_frames),
            "max_gap_frames": int(args.max_gap_frames),
            "w_iou": float(args.w_iou),
            "w_dist": float(args.w_dist),
            "w_helper": float(args.w_helper),
            "helper_class_id": int(args.helper_class_id),
            "use_helper_signal": not bool(args.disable_helper_signal),
        },
        "num_sequences": int(len(seqs)),
        "num_events": int(len(all_events)),
        "events_per_sequence": per_sequence,
        "events": all_events,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote Idea 2 events: {out_path}")
    print(f"Events: {len(all_events)} across {len(seqs)} sequences")


if __name__ == "__main__":
    main()
