#!/usr/bin/env python3
"""Run RF-DETR + ByteTrack on a video and render IDs + trajectories."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow direct script execution (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_video_common import (
    color_for_id,
    color_for_state,
    state_from_class_id,
    state_priority_soft_relabel_xyxy,
    write_tracking_analytics,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _to_tracked_dicts(
    out: Any,
) -> list[dict[str, object]]:
    dets: list[dict[str, object]] = []
    xyxy = np.asarray(getattr(out, "xyxy", np.zeros((0, 4), dtype=np.float32)))
    confs = np.asarray(getattr(out, "confidence", np.zeros((0,), dtype=np.float32))).reshape(-1)
    clss = np.asarray(getattr(out, "class_id", np.zeros((0,), dtype=np.int32))).reshape(-1)
    tids = np.asarray(getattr(out, "tracker_id", np.zeros((0,), dtype=np.int32))).reshape(-1)
    for i in range(xyxy.shape[0]):
        tid = int(tids[i]) if i < len(tids) else -1
        if tid < 0:
            continue
        dets.append(
            {
                "xyxy": [float(v) for v in xyxy[i].tolist()],
                "track_id": tid,
                "class_id": int(clss[i]) if i < len(clss) else 0,
                "score": float(confs[i]) if i < len(confs) else 0.0,
            }
        )
    return dets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True, help="Path to RF-DETR checkpoint (.pth)")
    p.add_argument("--source-video", type=str, required=True, help="Input video path")
    p.add_argument("--out-video", type=str, required=True, help="Output video path")
    p.add_argument("--model-class", type=str, default="RFDETRSmall", help="RF-DETR class name")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--trail-len", type=int, default=30, help="Max history points per track")
    p.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Thickness for boxes and trajectory lines",
    )
    p.add_argument(
        "--font-scale",
        type=float,
        default=0.9,
        help="Label font scale (increase for bigger text).",
    )
    p.add_argument(
        "--label-thickness",
        type=int,
        default=2,
        help="Label text thickness.",
    )
    p.add_argument(
        "--color-mode",
        choices=("state", "id"),
        default="state",
        help="Overlay color scheme: state-only (normal/trophallaxis) or per-id.",
    )
    p.add_argument(
        "--analytics-out",
        type=str,
        default=None,
        help="Optional JSON path to write tracking analytics.",
    )
    p.add_argument(
        "--state-priority-soft",
        action="store_true",
        help=(
            "Soft state preference: relabel normal->trophallaxis when overlap is high "
            "and score gap is small (no box deletion)."
        ),
    )
    p.add_argument(
        "--state-priority-iou-thresh",
        type=float,
        default=0.7,
        help="IoU threshold for --state-priority-soft.",
    )
    p.add_argument(
        "--state-priority-score-gap-max",
        type=float,
        default=0.12,
        help="Max (trophallaxis_score - normal_score) to allow soft relabel.",
    )
    p.add_argument("--track-thresh", type=float, default=0.25, help="ByteTrack activation threshold")
    p.add_argument("--match-thresh", type=float, default=0.8, help="ByteTrack match threshold")
    p.add_argument("--track-buffer", type=int, default=30, help="ByteTrack lost-track buffer")
    p.add_argument(
        "--optimize-for-inference",
        action="store_true",
        help="Call RF-DETR optimize_for_inference() before processing video.",
    )
    args = p.parse_args()

    try:
        import supervision as sv
    except ImportError:
        print("Install supervision first (pip install supervision).", file=sys.stderr)
        sys.exit(1)

    try:
        rfdetr = __import__("rfdetr")
    except ImportError:
        print("Install rfdetr first (pip install rfdetr).", file=sys.stderr)
        sys.exit(1)

    if not hasattr(rfdetr, args.model_class):
        print(f"Unknown model class: {args.model_class}", file=sys.stderr)
        sys.exit(1)

    ModelCls = getattr(rfdetr, args.model_class)

    weights = Path(args.weights).expanduser().resolve()
    source_video = Path(args.source_video).expanduser().resolve()
    out_video = Path(args.out_video).expanduser().resolve()
    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)
    if not source_video.is_file():
        print(f"Video not found: {source_video}", file=sys.stderr)
        sys.exit(1)

    model = ModelCls(pretrain_weights=str(weights))
    if args.optimize_for_inference:
        opt_fn = getattr(model, "optimize_for_inference", None)
        if callable(opt_fn):
            opt_fn()

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        print(f"Failed to open video: {source_video}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        print(f"Failed to read video dimensions: {source_video}", file=sys.stderr)
        sys.exit(1)

    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        print(f"Failed to open output writer: {out_video}", file=sys.stderr)
        sys.exit(1)

    tracker = sv.ByteTrack(
        track_activation_threshold=float(args.track_thresh),
        minimum_matching_threshold=float(args.match_thresh),
        lost_track_buffer=int(args.track_buffer),
    )

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=max(1, int(args.trail_len)))
    )
    state_counts: dict[str, int] = defaultdict(int)
    track_frames: dict[int, list[int]] = defaultdict(list)
    soft_relabels = 0
    frame_index = -1
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw = model.predict(rgb, threshold=float(args.conf))
            xyxy = np.asarray(getattr(raw, "xyxy", np.zeros((0, 4), dtype=np.float32)))
            conf = np.asarray(getattr(raw, "confidence", np.zeros((0,), dtype=np.float32))).reshape(-1)
            cls = np.asarray(getattr(raw, "class_id", np.zeros((0,), dtype=np.int32))).reshape(-1)

            dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
            tracked = tracker.update_with_detections(dets)
            out_dets = _to_tracked_dicts(tracked)
            if args.state_priority_soft:
                out_dets, n = state_priority_soft_relabel_xyxy(
                    out_dets,
                    iou_thresh=float(args.state_priority_iou_thresh),
                    score_gap_max=float(args.state_priority_score_gap_max),
                )
                soft_relabels += int(n)

            for d in out_dets:
                tid = int(d["track_id"])
                x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                trails[tid].append((cx, cy))
                c = int(d["class_id"])
                state = state_from_class_id(c)
                color = color_for_state(state) if args.color_mode == "state" else color_for_id(tid)
                state_counts[state] += 1
                track_frames[tid].append(frame_index)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, int(args.line_thickness)))
                s = float(d["score"])
                label = f"id:{tid} state:{state} {s:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    float(args.font_scale),
                    color,
                    max(1, int(args.label_thickness)),
                    cv2.LINE_AA,
                )

                pts = list(trails[tid])
                for j in range(1, len(pts)):
                    cv2.line(
                        frame,
                        pts[j - 1],
                        pts[j],
                        color,
                        max(1, int(args.line_thickness)),
                        cv2.LINE_AA,
                    )

            writer.write(frame)
            frames += 1
    finally:
        cap.release()
        writer.release()

    print(f"Wrote tracked video: {out_video}")
    print(f"Frames processed: {frames}")

    if args.analytics_out:
        ap = Path(args.analytics_out).expanduser().resolve()
        write_tracking_analytics(
            analytics_out=ap,
            source_video=source_video,
            output_video=out_video,
            repo_root=_REPO_ROOT,
            frames=frames,
            state_counts=dict(state_counts),
            track_frames={k: list(v) for k, v in track_frames.items()},
            soft_relabels=soft_relabels,
            tracker_info={
                "tracker": "bytetrack",
                "with_reid": False,
                "conf": float(args.conf),
                "track_thresh": float(args.track_thresh),
                "match_thresh": float(args.match_thresh),
                "track_buffer": int(args.track_buffer),
            },
            state_priority_info={
                "enabled": bool(args.state_priority_soft),
                "iou_thresh": float(args.state_priority_iou_thresh),
                "score_gap_max": float(args.state_priority_score_gap_max),
            },
            extra={
                "backend": "rfdetr",
                "model_class": str(args.model_class),
                "optimize_for_inference": bool(args.optimize_for_inference),
            },
        )
        print(f"Wrote analytics: {ap}")


if __name__ == "__main__":
    main()
