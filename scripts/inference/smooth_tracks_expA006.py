#!/usr/bin/env python3
"""EXP-A006: temporal smoothing on tracked detections and COCO rebuild."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from expA006_temporal import (
    group_frames,
    load_coco_images,
    load_sequence_map_from_manifest,
    smooth_tracks,
    tracks_to_coco_predictions,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=str, required=True, help="COCO GT JSON")
    p.add_argument("--tracks", type=str, required=True, help="tracks JSON from ByteTrack stage")
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional sequence_map manifest JSON",
    )
    p.add_argument("--out-pred", type=str, required=True, help="Smoothed COCO predictions JSON")
    p.add_argument("--out-tracks", type=str, required=True, help="Smoothed tracks JSON")
    p.add_argument("--stats-out", type=str, required=True, help="Smoothing stats JSON")
    p.add_argument("--min-track-len", type=int, default=3)
    p.add_argument("--fill-gap-max", type=int, default=1)
    args = p.parse_args()

    gt_path = Path(args.gt).expanduser().resolve()
    tr_path = Path(args.tracks).expanduser().resolve()
    pred_out = Path(args.out_pred).expanduser().resolve()
    tracks_out = Path(args.out_tracks).expanduser().resolve()
    stats_out = Path(args.stats_out).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None

    for path in (gt_path, tr_path):
        if not path.is_file():
            print(f"Missing required file: {path}", file=sys.stderr)
            sys.exit(1)

    images = load_coco_images(gt_path)
    seq_map = load_sequence_map_from_manifest(manifest_path)
    grouped, _, pos_to_img = group_frames(images, seq_map)

    tr_raw = json.loads(tr_path.read_text(encoding="utf-8"))
    tr_list = tr_raw.get("tracks") if isinstance(tr_raw, dict) else tr_raw
    if not isinstance(tr_list, list):
        print("tracks file must contain a list or {'tracks': [...]} payload", file=sys.stderr)
        sys.exit(1)

    smoothed_tracks, stats = smooth_tracks(
        tracks=tr_list,
        pos_to_img=pos_to_img,
        min_track_len=int(args.min_track_len),
        fill_gap_max=int(args.fill_gap_max),
    )
    stats["n_sequences"] = len(grouped)

    smoothed_pred = tracks_to_coco_predictions(smoothed_tracks)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    pred_out.write_text(json.dumps(smoothed_pred, indent=2), encoding="utf-8")

    tracks_payload = {
        "tracks": smoothed_tracks,
        "smoothing_config": {
            "min_track_len": int(args.min_track_len),
            "fill_gap_max": int(args.fill_gap_max),
            "score_mode": "track_average",
        },
    }
    tracks_out.parent.mkdir(parents=True, exist_ok=True)
    tracks_out.write_text(json.dumps(tracks_payload, indent=2), encoding="utf-8")

    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Wrote smoothed predictions: {pred_out} ({len(smoothed_pred)} detections)")
    print(f"Wrote smoothed tracks: {tracks_out}")
    print(f"Wrote smoothing stats: {stats_out}")


if __name__ == "__main__":
    main()
