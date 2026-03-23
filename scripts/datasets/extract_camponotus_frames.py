#!/usr/bin/env python3
"""Extract ordered image frames from videos for Camponotus sequences."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


@dataclass
class VideoSummary:
    video: str
    sequence: str
    fps_input: float
    fps_output: float
    frames_written: int
    first_index: int
    last_index: int


def _video_files(root: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    out = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    return out


def _extract_one(
    video_path: Path,
    out_seq_dir: Path,
    out_fps: float,
    start_s: float,
    end_s: float | None,
    image_ext: str,
) -> VideoSummary:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if in_fps <= 0:
        in_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_frame = max(0, int(round(start_s * in_fps)))
    if end_s is None:
        end_frame = max(0, total_frames - 1) if total_frames > 0 else 10**12
    else:
        end_frame = max(start_frame, int(round(end_s * in_fps)))

    frame_step = max(1, int(round(in_fps / out_fps)))

    out_seq_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    next_src_idx = start_frame
    written = 0

    # Continue index if directory already has frames.
    existing = sorted(p for p in out_seq_dir.glob("*") if p.suffix.lower() == image_ext)
    if existing:
        try:
            frame_idx = int(existing[-1].stem)
        except ValueError:
            frame_idx = len(existing)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur_src = start_frame
    while cur_src <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if cur_src >= next_src_idx:
            frame_idx += 1
            out_name = f"{frame_idx:06d}{image_ext}"
            out_path = out_seq_dir / out_name
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"Failed to write frame: {out_path}")
            written += 1
            next_src_idx += frame_step
        cur_src += 1

    cap.release()
    first_index = frame_idx - written + 1 if written > 0 else frame_idx
    return VideoSummary(
        video=str(video_path),
        sequence=out_seq_dir.name,
        fps_input=in_fps,
        fps_output=out_fps,
        frames_written=written,
        first_index=first_index,
        last_index=frame_idx,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--videos-root",
        type=str,
        required=True,
        help="Directory containing source videos (recursively searched).",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="datasets/camponotus_raw/in_situ",
        help="Output root for extracted sequences.",
    )
    p.add_argument("--fps", type=float, default=2.0, help="Target extraction FPS.")
    p.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds.")
    p.add_argument("--end-sec", type=float, default=None, help="Optional end time in seconds.")
    p.add_argument(
        "--seq-prefix",
        type=str,
        default="seq_",
        help="Output sequence prefix (example: seq_001).",
    )
    p.add_argument(
        "--image-ext",
        type=str,
        default=".jpg",
        help="Output image extension (example: .jpg).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without writing files.",
    )
    args = p.parse_args()

    videos_root = Path(args.videos_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    image_ext = args.image_ext if args.image_ext.startswith(".") else f".{args.image_ext}"
    fps = float(args.fps)
    if fps <= 0:
        print("--fps must be > 0", file=sys.stderr)
        sys.exit(1)
    if not videos_root.is_dir():
        print(f"videos root not found: {videos_root}", file=sys.stderr)
        sys.exit(1)

    vids = _video_files(videos_root)
    if not vids:
        print(f"No videos found under {videos_root}", file=sys.stderr)
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)
    summaries: list[VideoSummary] = []
    for i, vp in enumerate(vids, start=1):
        seq_name = f"{args.seq_prefix}{i:03d}"
        out_seq = out_root / seq_name
        if args.dry_run:
            print(f"[dry-run] {vp} -> {out_seq}")
            continue
        sm = _extract_one(
            video_path=vp,
            out_seq_dir=out_seq,
            out_fps=fps,
            start_s=float(args.start_sec),
            end_s=None if args.end_sec is None else float(args.end_sec),
            image_ext=image_ext,
        )
        summaries.append(sm)
        print(f"{sm.video} -> {out_seq} ({sm.frames_written} frames)")

    if args.dry_run:
        print(f"[dry-run] {len(vids)} videos discovered")
        return

    manifest: dict[str, Any] = {
        "videos_root": str(videos_root),
        "out_root": str(out_root),
        "fps": fps,
        "start_sec": float(args.start_sec),
        "end_sec": None if args.end_sec is None else float(args.end_sec),
        "image_ext": image_ext,
        "videos_processed": len(summaries),
        "total_frames_written": int(sum(s.frames_written for s in summaries)),
        "sequences": [s.__dict__ for s in summaries],
    }
    mf = out_root / "frame_extraction_manifest.json"
    mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {mf}")


if __name__ == "__main__":
    main()
