#!/usr/bin/env python3
"""Batch-run Camponotus video autolabeling for a folder of videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _collect_videos(videos_root: Path, recursive: bool) -> list[Path]:
    iterator = videos_root.rglob("*") if recursive else videos_root.glob("*")
    videos = [p for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    videos.sort(key=lambda p: str(p))
    return videos


def _safe_name(video_path: Path) -> str:
    stem = video_path.stem
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)


def _build_cmd(args: argparse.Namespace, video_path: Path, out_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parents[1] / "datasets" / "bootstrap_camponotus_autolabel.py").resolve()),
        "--source-video",
        str(video_path.resolve()),
        "--backend",
        "yolo",
        "--yolo-weights",
        str(Path(args.yolo_weights).expanduser().resolve()),
        "--conf",
        str(float(args.conf)),
        "--with-tracking",
        "--tracker",
        str(args.tracker),
        "--track-thresh",
        str(float(args.track_thresh)),
        "--match-thresh",
        str(float(args.match_thresh)),
        "--track-buffer",
        str(int(args.track_buffer)),
        "--min-track-len",
        str(int(args.min_track_len)),
        "--out",
        str(out_json.resolve()),
    ]
    if args.yolo_imgsz is not None:
        cmd.extend(["--yolo-imgsz", str(int(args.yolo_imgsz))])
    if args.tracker == "botsort" and args.botsort_with_reid:
        cmd.append("--botsort-with-reid")
    if args.tracker == "botsort":
        cmd.extend(
            [
                "--botsort-proximity-thresh",
                str(float(args.botsort_proximity_thresh)),
                "--botsort-appearance-thresh",
                str(float(args.botsort_appearance_thresh)),
            ]
        )
    if args.state_priority_soft:
        cmd.extend(
            [
                "--state-priority-soft",
                "--state-priority-iou-thresh",
                str(float(args.state_priority_iou_thresh)),
                "--state-priority-score-gap-max",
                str(float(args.state_priority_score_gap_max)),
            ]
        )
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos-root", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--yolo-weights", required=True, type=str)
    p.add_argument("--recursive", action="store_true", help="Recursively scan for videos.")
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--yolo-imgsz", type=int, default=None)
    p.add_argument("--tracker", choices=("bytetrack", "botsort"), default="botsort")
    p.add_argument("--track-thresh", type=float, default=0.25)
    p.add_argument("--match-thresh", type=float, default=0.8)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--min-track-len", type=int, default=2)
    p.add_argument("--botsort-with-reid", action="store_true")
    p.add_argument("--botsort-proximity-thresh", type=float, default=0.5)
    p.add_argument("--botsort-appearance-thresh", type=float, default=0.25)
    p.add_argument("--state-priority-soft", action="store_true")
    p.add_argument("--state-priority-iou-thresh", type=float, default=0.7)
    p.add_argument("--state-priority-score-gap-max", type=float, default=0.12)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    videos_root = Path(args.videos_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    weights = Path(args.yolo_weights).expanduser().resolve()
    if not videos_root.is_dir():
        raise SystemExit(f"videos root not found: {videos_root}")
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = _collect_videos(videos_root, recursive=bool(args.recursive))
    if not videos:
        raise SystemExit("no videos found")

    run_items: list[dict[str, Any]] = []
    failures = 0
    for idx, video in enumerate(videos, start=1):
        base = _safe_name(video)
        out_json = out_dir / f"{base}_tracked.json"
        cmd = _build_cmd(args, video, out_json)
        print(f"[{idx}/{len(videos)}] {video.name}")
        print("  " + " ".join(cmd))
        item: dict[str, Any] = {
            "video": str(video),
            "output_json": str(out_json),
            "returncode": None,
        }
        if not args.dry_run:
            proc = subprocess.run(cmd, check=False)
            item["returncode"] = int(proc.returncode)
            if proc.returncode != 0:
                failures += 1
        run_items.append(item)

    manifest = {
        "videos_root": str(videos_root),
        "out_dir": str(out_dir),
        "total_videos": len(videos),
        "failed_videos": failures,
        "items": run_items,
    }
    manifest_path = out_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")

    if failures > 0:
        raise SystemExit(f"completed with failures: {failures}/{len(videos)}")
    print("Batch prelabeling completed.")


if __name__ == "__main__":
    main()
