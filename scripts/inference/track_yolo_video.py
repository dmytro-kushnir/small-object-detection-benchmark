#!/usr/bin/env python3
"""Run YOLO tracking on video and render IDs + trajectory lines."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2

# Allow direct script execution (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_video_common import (
    color_for_id,
    color_for_state,
    state_from_class_id,
    state_priority_soft_relabel_xyxy,
    write_tracking_analytics,
)
from yolo_track_common import (
    build_tracker_config,
    iter_tracked_detections,
    temporary_tracker_yaml,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True, help="Path to YOLO .pt weights")
    p.add_argument("--source-video", type=str, required=True, help="Input video path")
    p.add_argument("--out-video", type=str, required=True, help="Output video path")
    p.add_argument("--tracker", choices=("bytetrack", "botsort"), default="botsort")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=None, help="Optional YOLO tracking imgsz")
    p.add_argument("--device", type=str, default=None, help='Device, e.g. "0" or "cpu"')
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
    p.add_argument("--track-thresh", type=float, default=0.25, help="Tracker high threshold")
    p.add_argument("--match-thresh", type=float, default=0.8, help="Tracker match threshold")
    p.add_argument("--track-buffer", type=int, default=30, help="Tracker lost-track buffer")
    p.add_argument(
        "--botsort-with-reid",
        action="store_true",
        help="Enable BoT-SORT appearance ReID.",
    )
    p.add_argument(
        "--botsort-proximity-thresh",
        type=float,
        default=0.5,
        help="BoT-SORT proximity threshold.",
    )
    p.add_argument(
        "--botsort-appearance-thresh",
        type=float,
        default=0.25,
        help="BoT-SORT appearance threshold.",
    )
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics first (see requirements.txt).", file=sys.stderr)
        sys.exit(1)

    weights = Path(args.weights).expanduser().resolve()
    source_video = Path(args.source_video).expanduser().resolve()
    out_video = Path(args.out_video).expanduser().resolve()
    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)
    if not source_video.is_file():
        print(f"Video not found: {source_video}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        print(f"Failed to open video: {source_video}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if width <= 0 or height <= 0:
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
        print(f"Failed to open output writer: {out_video}", file=sys.stderr)
        sys.exit(1)

    tracker_cfg = build_tracker_config(
        tracker=str(args.tracker),
        track_thresh=float(args.track_thresh),
        match_thresh=float(args.match_thresh),
        track_buffer=int(args.track_buffer),
        with_reid=bool(args.botsort_with_reid),
        proximity_thresh=float(args.botsort_proximity_thresh),
        appearance_thresh=float(args.botsort_appearance_thresh),
    )

    model = YOLO(str(weights))

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=max(1, int(args.trail_len)))
    )
    state_counts: dict[str, int] = defaultdict(int)
    track_frames: dict[int, list[int]] = defaultdict(list)
    soft_relabels = 0
    frame_index = -1
    frames = 0
    try:
        with temporary_tracker_yaml(tracker_cfg, suffix=f"_{args.tracker}.yaml") as tracker_cfg_path:
            tracked_iter = iter_tracked_detections(
                model,
                source=str(source_video),
                conf=float(args.conf),
                tracker_cfg_path=tracker_cfg_path,
                imgsz=int(args.imgsz) if args.imgsz is not None else None,
                device=str(args.device) if args.device is not None else None,
                persist=True,
                stream=True,
                verbose=False,
            )
            for r, dets in tracked_iter:
                frame = r.orig_img.copy() if r.orig_img is not None else None
                if frame is None:
                    continue
                frame_index += 1
                if args.state_priority_soft:
                    dets, n = _state_priority_soft_relabel_xyxy(
                        dets,
                        iou_thresh=float(args.state_priority_iou_thresh),
                        score_gap_max=float(args.state_priority_score_gap_max),
                    )
                    soft_relabels += int(n)

                for d in dets:
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
                "tracker": str(args.tracker),
                "with_reid": bool(args.botsort_with_reid),
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
            extra=None,
        )
        print(f"Wrote analytics: {ap}")


if __name__ == "__main__":
    main()
