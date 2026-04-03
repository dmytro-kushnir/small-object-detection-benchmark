#!/usr/bin/env python3
"""Shared helpers for video tracking visualization scripts."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from repo_paths import path_for_artifact


def color_for_id(track_id: int) -> tuple[int, int, int]:
    return (37 * track_id % 255, 97 * track_id % 255, 173 * track_id % 255)


def state_from_class_id(class_id: int) -> str:
    return "trophallaxis" if int(class_id) == 1 else "normal"


def color_for_state(state: str) -> tuple[int, int, int]:
    # BGR: green for normal, red for trophallaxis
    return (0, 220, 0) if state == "normal" else (0, 0, 230)


def bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def state_priority_soft_relabel_xyxy(
    dets: list[dict[str, object]],
    *,
    iou_thresh: float,
    score_gap_max: float,
) -> tuple[list[dict[str, object]], int]:
    troph = [d for d in dets if int(d.get("class_id", -1)) == 1]
    if not troph:
        return dets, 0
    out: list[dict[str, object]] = []
    relabeled = 0
    for d in dets:
        if int(d.get("class_id", -1)) != 0:
            out.append(d)
            continue
        n_score = float(d.get("score", 0.0))
        should_flip = False
        for t in troph:
            iou = bbox_iou_xyxy(
                [float(v) for v in d["xyxy"]],  # type: ignore[index]
                [float(v) for v in t["xyxy"]],  # type: ignore[index]
            )
            if iou < float(iou_thresh):
                continue
            score_gap = float(t.get("score", 0.0)) - n_score
            if score_gap >= 0.0 and score_gap <= float(score_gap_max):
                should_flip = True
                break
        if should_flip:
            nd = dict(d)
            nd["class_id"] = 1
            out.append(nd)
            relabeled += 1
        else:
            out.append(d)
    return out, relabeled


def state_priority_consensus_relabel_normal_near_troph_xyxy(
    dets: list[dict[str, object]],
    *,
    iou_thresh: float,
) -> tuple[list[dict[str, object]], int]:
    """Relabel normal→trophallaxis if the box overlaps any trophallaxis detection.

    Unlike :func:`state_priority_soft_relabel_xyxy`, there is **no** score-gap check:
    any high-IoU pair forces the normal side to trophallaxis. Intended for **OOD /
    visualization**; can increase false trophallaxis when unrelated ants touch.
    """
    troph = [d for d in dets if int(d.get("class_id", -1)) == 1]
    if not troph:
        return dets, 0
    out: list[dict[str, object]] = []
    relabeled = 0
    thr = float(iou_thresh)
    for d in dets:
        if int(d.get("class_id", -1)) != 0:
            out.append(d)
            continue
        n_score = float(d.get("score", 0.0))
        partner_score = 0.0
        should_flip = False
        for t in troph:
            iou = bbox_iou_xyxy(
                [float(v) for v in d["xyxy"]],  # type: ignore[index]
                [float(v) for v in t["xyxy"]],  # type: ignore[index]
            )
            if iou < thr:
                continue
            should_flip = True
            partner_score = max(partner_score, float(t.get("score", 0.0)))
        if should_flip:
            nd = dict(d)
            nd["class_id"] = 1
            nd["score"] = max(n_score, partner_score)
            out.append(nd)
            relabeled += 1
        else:
            out.append(d)
    return out, relabeled


def temporal_majority_smooth_dets(
    dets: list[dict[str, object]],
    *,
    history: dict[int, deque[int]],
) -> list[dict[str, object]]:
    """Sliding-window majority vote on ``class_id`` (0/1) per ``track_id``.

    Append each frame's class (typically after state-priority relabel steps)
    to ``history[track_id]``. Deques must use ``maxlen=K`` set by the caller.

    **Tie rule:** if normal and trophallaxis counts are equal in the window, keep
    this frame's class (no bias toward either label).

    Returns shallow-copied det dicts with updated ``class_id``.
    """
    out: list[dict[str, object]] = []
    for d in dets:
        tid = int(d["track_id"])
        cid = int(d.get("class_id", 0))
        q = history[tid]
        q.append(cid)
        c0 = sum(1 for x in q if int(x) == 0)
        c1 = len(q) - c0
        if c0 == c1:
            smoothed = cid
        elif c0 > c1:
            smoothed = 0
        else:
            smoothed = 1
        nd = dict(d)
        nd["class_id"] = int(smoothed)
        out.append(nd)
    return out


def _per_track_state_summary(
    track_state_counts: dict[int, dict[str, int]],
    track_frames: dict[int, list[int]],
) -> dict[str, Any]:
    """Per-track class histogram + dominant state (helps debug OOD / mis-trophallaxis on one identity)."""
    out: dict[str, Any] = {}
    for tid in sorted(track_state_counts.keys()):
        counts = {k: int(v) for k, v in track_state_counts[tid].items()}
        total = sum(counts.values())
        dom = max(counts, key=counts.get) if counts else "unknown"
        frac = float(counts.get(dom, 0)) / float(total) if total else 0.0
        out[str(tid)] = {
            "state_counts": counts,
            "dominant_state": dom,
            "dominant_fraction": round(frac, 4),
            "detection_observations": int(total),
            "frame_indices_recorded": int(len(track_frames.get(tid, []))),
        }
    return {"per_track": out}


def write_tracking_analytics(
    *,
    analytics_out: Path,
    source_video: Path,
    output_video: Path,
    repo_root: Path,
    frames: int,
    state_counts: dict[str, int],
    track_frames: dict[int, list[int]],
    soft_relabels: int,
    tracker_info: dict[str, Any],
    state_priority_info: dict[str, Any],
    state_priority_consensus_info: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    track_state_counts: dict[int, dict[str, int]] | None = None,
) -> None:
    lens = [len(v) for v in track_frames.values()]
    short2 = sum(1 for l in lens if l <= 2)
    short3 = sum(1 for l in lens if l <= 3)
    gap_events = 0
    gap_frames_total = 0
    for fs in track_frames.values():
        s = sorted(fs)
        for i in range(1, len(s)):
            gap = s[i] - s[i - 1] - 1
            if gap > 0:
                gap_events += 1
                gap_frames_total += gap
    analytics: dict[str, Any] = {
        "source_video": path_for_artifact(source_video, repo_root),
        "output_video": path_for_artifact(output_video, repo_root),
        "frames_processed": int(frames),
        "states": dict(state_counts),
        "unique_tracks": int(len(track_frames)),
        "track_len_mean": float(np.mean(lens)) if lens else 0.0,
        "track_len_median": float(np.median(lens)) if lens else 0.0,
        "short_tracks_len_le_2": int(short2),
        "short_tracks_len_le_3": int(short3),
        "gap_events": int(gap_events),
        "gap_frames_total": int(gap_frames_total),
        "state_priority_soft": {
            **state_priority_info,
            "relabel_count": int(soft_relabels),
        },
    }
    if state_priority_consensus_info is not None:
        analytics["state_priority_consensus"] = dict(state_priority_consensus_info)
    analytics.update(tracker_info)
    if track_state_counts:
        analytics.update(_per_track_state_summary(track_state_counts, track_frames))
    if extra:
        analytics.update(extra)

    analytics_out.parent.mkdir(parents=True, exist_ok=True)
    analytics_out.write_text(json.dumps(analytics, indent=2), encoding="utf-8")
