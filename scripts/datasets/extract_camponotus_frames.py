#!/usr/bin/env python3
"""Extract ordered image frames from videos for Camponotus sequences."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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
    rotation_deg: int


def _video_files(root: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    out = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    return out


def _sanitize_video_stem(video_path: Path, max_stem_len: int = 200) -> str:
    """Lowercase slug from filename stem for use in seq_* directory names."""
    stem = video_path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "unknown"
    if len(stem) > max_stem_len:
        stem = stem[:max_stem_len].rstrip("_")
    return stem


def _effective_output_fps(
    video_path: Path,
    default_fps: float,
    troph_fps: float | None,
    troph_substring: str,
) -> float:
    """Use troph_fps when filename stem matches substring (case-insensitive)."""
    if troph_fps is None or troph_fps <= 0:
        return default_fps
    sub = troph_substring.lower().strip()
    if not sub:
        return default_fps
    if sub in video_path.stem.lower():
        return troph_fps
    return default_fps


def _allocate_seq_dir_name(
    out_root: Path,
    video_path: Path,
    used_names: set[str],
    *,
    naming: str,
    seq_prefix: str,
    index_one_based: int,
) -> str:
    """
    Return directory name (not full path) under out_root.
    naming: 'video' -> seq_<slug> with collision suffix; 'index' -> seq_prefix + zero-padded index.
    """
    if naming == "index":
        name = f"{seq_prefix}{index_one_based:03d}"
        used_names.add(name)
        return name

    base = _sanitize_video_stem(video_path)
    candidate = f"{seq_prefix}{base}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    n = 2
    while True:
        cand = f"{seq_prefix}{base}_{n}"
        if cand not in used_names:
            used_names.add(cand)
            return cand
        n += 1


def _probe_rotation_deg(video_path: Path) -> int:
    """Read clockwise rotation from ffprobe metadata if present."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream_tags=rotate",
        "-of",
        "default=noprint_wrappers=1",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return 0
    if proc.returncode != 0:
        return 0
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("TAG:rotate="):
            continue
        try:
            deg = int(round(float(line.split("=", 1)[1])))
            return deg % 360
        except (TypeError, ValueError):
            continue
    return 0


def _max_frame_index_from_existing(
    existing: list[Path],
    *,
    seq_dir_name: str,
    unique_frame_basenames: bool,
) -> int:
    """Largest numeric frame index already on disk (0 if none / unparsable)."""
    if not existing:
        return 0
    paths = sorted(existing)
    if not unique_frame_basenames:
        try:
            return int(paths[-1].stem)
        except ValueError:
            return len(paths)
    prefix = f"{seq_dir_name}_"
    max_i = 0
    for p in paths:
        stem = p.stem
        if not stem.startswith(prefix):
            continue
        tail = stem[len(prefix) :]
        try:
            max_i = max(max_i, int(tail, 10))
        except ValueError:
            continue
    return max_i


def _frame_output_name(
    seq_dir_name: str,
    frame_idx: int,
    image_ext: str,
    *,
    unique_frame_basenames: bool,
) -> str:
    if unique_frame_basenames:
        return f"{seq_dir_name}_{frame_idx:06d}{image_ext}"
    return f"{frame_idx:06d}{image_ext}"


def _apply_rotation(frame: Any, rotation_deg: int) -> Any:
    """Rotate frame clockwise in 90-degree steps."""
    rot = rotation_deg % 360
    if rot == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _extract_one(
    video_path: Path,
    out_seq_dir: Path,
    out_fps: float,
    start_s: float,
    end_s: float | None,
    image_ext: str,
    apply_rotation: bool,
    clean_seq_dir: bool,
    *,
    unique_frame_basenames: bool,
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
    rotation_deg = _probe_rotation_deg(video_path) if apply_rotation else 0

    out_seq_dir.mkdir(parents=True, exist_ok=True)
    seq_dir_name = out_seq_dir.name
    frame_idx = 0
    next_src_idx = start_frame
    written = 0

    # On reruns, default to replacing previous extraction in this sequence.
    existing = sorted(p for p in out_seq_dir.glob("*") if p.suffix.lower() == image_ext)
    if existing and clean_seq_dir:
        for p in existing:
            p.unlink()
        existing = []

    # Optional append mode for incremental extraction.
    if existing and not clean_seq_dir:
        frame_idx = _max_frame_index_from_existing(
            existing,
            seq_dir_name=seq_dir_name,
            unique_frame_basenames=unique_frame_basenames,
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur_src = start_frame
    while cur_src <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if cur_src >= next_src_idx:
            frame_idx += 1
            out_name = _frame_output_name(
                seq_dir_name,
                frame_idx,
                image_ext,
                unique_frame_basenames=unique_frame_basenames,
            )
            out_path = out_seq_dir / out_name
            frame = _apply_rotation(frame, rotation_deg)
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
        rotation_deg=int(rotation_deg),
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
    p.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target extraction FPS for most videos (default: 2).",
    )
    p.add_argument(
        "--fps-trophallaxis",
        type=float,
        default=None,
        help=(
            "If set, videos whose filename contains --trophallaxis-substring "
            "(default: trophallaxis) use this FPS instead of --fps."
        ),
    )
    p.add_argument(
        "--trophallaxis-substring",
        type=str,
        default="trophallaxis",
        help="Case-insensitive match against video filename stem (default: trophallaxis).",
    )
    p.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds.")
    p.add_argument("--end-sec", type=float, default=None, help="Optional end time in seconds.")
    p.add_argument(
        "--seq-prefix",
        type=str,
        default="seq_",
        help="Sequence folder prefix. With --seq-naming video: seq_<slug> (e.g. seq_camponotus_003).",
    )
    p.add_argument(
        "--seq-naming",
        choices=("video", "index"),
        default="video",
        help="video: folder from sanitized filename (default). index: seq_prefix + 001, 002, ...",
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
    p.add_argument(
        "--apply-rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply video rotation metadata (default: true).",
    )
    p.add_argument(
        "--clean-on-rerun",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove existing extracted frames in each sequence before writing (default: true).",
    )
    p.add_argument(
        "--unique-frame-basenames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Name frames seq_<slug>_NNNNNN.jpg inside each seq_* folder so basenames are unique "
            "repo-wide. Use when CVAT exports flat file_name (required for "
            "align_coco_filenames_to_camponotus_raw.py with manifest splits)."
        ),
    )
    args = p.parse_args()

    videos_root = Path(args.videos_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    image_ext = args.image_ext if args.image_ext.startswith(".") else f".{args.image_ext}"
    fps = float(args.fps)
    if fps <= 0:
        print("--fps must be > 0", file=sys.stderr)
        sys.exit(1)
    troph_fps: float | None = None
    if args.fps_trophallaxis is not None:
        troph_fps = float(args.fps_trophallaxis)
        if troph_fps <= 0:
            print("--fps-trophallaxis must be > 0 when set", file=sys.stderr)
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
    used_seq_names: set[str] = set()
    for i, vp in enumerate(vids, start=1):
        seq_name = _allocate_seq_dir_name(
            out_root,
            vp,
            used_seq_names,
            naming=str(args.seq_naming),
            seq_prefix=str(args.seq_prefix),
            index_one_based=i,
        )
        out_seq = out_root / seq_name
        out_fps = _effective_output_fps(
            vp,
            fps,
            troph_fps,
            str(args.trophallaxis_substring),
        )
        if args.dry_run:
            print(f"[dry-run] {vp} -> {out_seq} (fps={out_fps})")
            continue
        sm = _extract_one(
            video_path=vp,
            out_seq_dir=out_seq,
            out_fps=out_fps,
            start_s=float(args.start_sec),
            end_s=None if args.end_sec is None else float(args.end_sec),
            image_ext=image_ext,
            apply_rotation=bool(args.apply_rotation),
            clean_seq_dir=bool(args.clean_on_rerun),
            unique_frame_basenames=bool(args.unique_frame_basenames),
        )
        summaries.append(sm)
        print(
            f"{sm.video} -> {out_seq} ({sm.frames_written} frames, "
            f"fps_out={sm.fps_output}, rotation={sm.rotation_deg}deg)"
        )

    if args.dry_run:
        print(f"[dry-run] {len(vids)} videos discovered")
        return

    manifest: dict[str, Any] = {
        "videos_root": str(videos_root),
        "out_root": str(out_root),
        "fps": fps,
        "fps_trophallaxis": troph_fps,
        "trophallaxis_substring": str(args.trophallaxis_substring),
        "start_sec": float(args.start_sec),
        "end_sec": None if args.end_sec is None else float(args.end_sec),
        "image_ext": image_ext,
        "apply_rotation": bool(args.apply_rotation),
        "seq_naming": str(args.seq_naming),
        "seq_prefix": str(args.seq_prefix),
        "unique_frame_basenames": bool(args.unique_frame_basenames),
        "videos_processed": len(summaries),
        "total_frames_written": int(sum(s.frames_written for s in summaries)),
        "sequences": [s.__dict__ for s in summaries],
    }
    mf = out_root / "frame_extraction_manifest.json"
    mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {mf}")


if __name__ == "__main__":
    main()
