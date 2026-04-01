#!/usr/bin/env python3
"""Evaluate Camponotus Idea 2 event predictions with temporal IoU matching."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _event_key(e: dict[str, Any]) -> tuple[str, tuple[int, int]]:
    return str(e["sequence_name"]), _pair_key(int(e["track_id_a"]), int(e["track_id_b"]))


def _tiou(a: dict[str, Any], b: dict[str, Any]) -> float:
    a0, a1 = int(a["start_frame"]), int(a["end_frame"])
    b0, b1 = int(b["start_frame"]), int(b["end_frame"])
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    if inter <= 0:
        return 0.0
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return float(inter / union) if union > 0 else 0.0


def _flatten_gt(gt: dict[str, Any]) -> list[dict[str, Any]]:
    clips = gt.get("clips", [])
    out: list[dict[str, Any]] = []
    for clip in clips:
        seq = str(clip.get("sequence_name", clip.get("clip_id", "unknown")))
        for e in clip.get("events", []):
            out.append(
                {
                    "event_id": str(e.get("event_id", "")),
                    "sequence_name": seq,
                    "track_id_a": int(e["track_id_a"]),
                    "track_id_b": int(e["track_id_b"]),
                    "start_frame": int(e["start_frame"]),
                    "end_frame": int(e["end_frame"]),
                }
            )
    return out


def _match_events(
    gt_events: list[dict[str, Any]],
    pred_events: list[dict[str, Any]],
    match_tiou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    gt_by_key: dict[tuple[str, tuple[int, int]], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    pred_by_key: dict[tuple[str, tuple[int, int]], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for i, g in enumerate(gt_events):
        gt_by_key[_event_key(g)].append((i, g))
    for j, p in enumerate(pred_events):
        pred_by_key[_event_key(p)].append((j, p))

    matches: list[tuple[int, int, float]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for key in sorted(set(gt_by_key.keys()) | set(pred_by_key.keys())):
        gt_list = gt_by_key.get(key, [])
        pred_list = pred_by_key.get(key, [])
        candidates: list[tuple[float, int, int]] = []
        for i, g in gt_list:
            for j, p in pred_list:
                t = _tiou(g, p)
                if t >= match_tiou_threshold:
                    candidates.append((t, i, j))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for t, i, j in candidates:
            if i in matched_gt or j in matched_pred:
                continue
            matched_gt.add(i)
            matched_pred.add(j)
            matches.append((i, j, t))
    return matches, matched_gt, matched_pred


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-events", type=str, required=True, help="GT benchmark JSON (clips/events format).")
    p.add_argument("--pred-events", type=str, required=True, help="Predicted events JSON from infer_camponotus_idea2_events.py.")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--match-tiou-threshold", type=float, default=0.30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt_events).expanduser().resolve()
    pred_path = Path(args.pred_events).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    gt_payload = json.loads(gt_path.read_text(encoding="utf-8"))
    pred_payload = json.loads(pred_path.read_text(encoding="utf-8"))
    gt_events = _flatten_gt(gt_payload)
    pred_events = list(pred_payload.get("events", []))

    matches, matched_gt, matched_pred = _match_events(
        gt_events=gt_events,
        pred_events=pred_events,
        match_tiou_threshold=float(args.match_tiou_threshold),
    )
    tp = len(matches)
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) else 0.0
    mean_tiou = _safe_div(sum(t for _, _, t in matches), len(matches))

    # Per-sequence report.
    per_seq_gt: dict[str, int] = defaultdict(int)
    per_seq_pred: dict[str, int] = defaultdict(int)
    per_seq_tp: dict[str, int] = defaultdict(int)
    per_seq_tious: dict[str, list[float]] = defaultdict(list)
    for g in gt_events:
        per_seq_gt[str(g["sequence_name"])] += 1
    for p in pred_events:
        per_seq_pred[str(p["sequence_name"])] += 1
    for gi, pj, t in matches:
        seq = str(gt_events[gi]["sequence_name"])
        per_seq_tp[seq] += 1
        per_seq_tious[seq].append(float(t))

    per_sequence = {}
    for seq in sorted(set(per_seq_gt) | set(per_seq_pred)):
        stp = int(per_seq_tp.get(seq, 0))
        sg = int(per_seq_gt.get(seq, 0))
        sp = int(per_seq_pred.get(seq, 0))
        sfp = sp - stp
        sfn = sg - stp
        sprec = _safe_div(stp, stp + sfp)
        srec = _safe_div(stp, stp + sfn)
        sf1 = _safe_div(2.0 * sprec * srec, sprec + srec) if (sprec + srec) else 0.0
        per_sequence[seq] = {
            "tp": stp,
            "fp": int(sfp),
            "fn": int(sfn),
            "precision": sprec,
            "recall": srec,
            "f1": sf1,
            "mean_tiou_matched": _safe_div(sum(per_seq_tious.get(seq, [])), len(per_seq_tious.get(seq, []))),
            "num_gt": sg,
            "num_pred": sp,
        }

    out = {
        "format": "camponotus_idea2_event_eval_v1",
        "gt_events": str(gt_path),
        "pred_events": str(pred_path),
        "match_tiou_threshold": float(args.match_tiou_threshold),
        "aggregate": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_tiou_matched": mean_tiou,
            "num_gt": int(len(gt_events)),
            "num_pred": int(len(pred_events)),
        },
        "per_sequence": per_sequence,
        "matches": [
            {
                "gt_index": int(gi),
                "pred_index": int(pj),
                "sequence_name": str(gt_events[gi]["sequence_name"]),
                "tiou": float(t),
            }
            for gi, pj, t in matches
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote Idea 2 event evaluation: {out_path}")
    print(f"TP/FP/FN: {tp}/{fp}/{fn} | F1={f1:.4f}")


if __name__ == "__main__":
    main()
