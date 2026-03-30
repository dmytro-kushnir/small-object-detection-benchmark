#!/usr/bin/env python3
"""Shared helpers for video tracking visualization scripts."""

from __future__ import annotations

import json
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
    extra: dict[str, Any] | None = None,
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
    analytics.update(tracker_info)
    if extra:
        analytics.update(extra)

    analytics_out.parent.mkdir(parents=True, exist_ok=True)
    analytics_out.write_text(json.dumps(analytics, indent=2), encoding="utf-8")
