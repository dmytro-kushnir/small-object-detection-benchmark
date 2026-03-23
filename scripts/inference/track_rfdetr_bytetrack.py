#!/usr/bin/env python3
"""EXP-A006: run ByteTrack on RF-DETR detections and save tracked records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from expA006_temporal import (
    group_frames,
    load_coco_images,
    load_predictions,
    load_sequence_map_from_manifest,
    run_bytetrack_on_predictions,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=str, required=True, help="COCO GT JSON")
    p.add_argument("--pred", type=str, required=True, help="COCO predictions JSON from RF-DETR")
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional sequence_map manifest JSON",
    )
    p.add_argument("--out", type=str, required=True, help="Tracked detections JSON")
    p.add_argument("--stats-out", type=str, required=True, help="Tracking stats JSON")
    p.add_argument("--track-thresh", type=float, default=0.25)
    p.add_argument("--match-thresh", type=float, default=0.8)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument(
        "--seg-filter-distance-abs",
        type=float,
        default=None,
        help="Optional absolute mask distance threshold for sv.filter_segments_by_distance.",
    )
    p.add_argument(
        "--seg-filter-distance-ratio",
        type=float,
        default=None,
        help="Optional relative mask distance threshold for sv.filter_segments_by_distance.",
    )
    args = p.parse_args()

    gt_path = Path(args.gt).expanduser().resolve()
    pred_path = Path(args.pred).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    stats_path = Path(args.stats_out).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None

    for path in (gt_path, pred_path):
        if not path.is_file():
            print(f"Missing required file: {path}", file=sys.stderr)
            sys.exit(1)

    images = load_coco_images(gt_path)
    seq_map = load_sequence_map_from_manifest(manifest_path)
    grouped, img_to_pos, _ = group_frames(images, seq_map)
    preds = load_predictions(pred_path)

    tracks, stats = run_bytetrack_on_predictions(
        preds=preds,
        grouped=grouped,
        img_to_pos=img_to_pos,
        track_thresh=float(args.track_thresh),
        match_thresh=float(args.match_thresh),
        track_buffer=int(args.track_buffer),
        seg_filter_distance_abs=args.seg_filter_distance_abs,
        seg_filter_distance_ratio=args.seg_filter_distance_ratio,
    )
    stats["n_sequences"] = len(grouped)
    stats["sequence_ids"] = sorted(grouped.keys())
    stats["source_pred_path"] = str(pred_path)

    out_payload = {
        "tracks": tracks,
        "tracking_config": {
            "track_thresh": float(args.track_thresh),
            "match_thresh": float(args.match_thresh),
            "track_buffer": int(args.track_buffer),
            "seg_filter_distance_abs": args.seg_filter_distance_abs,
            "seg_filter_distance_ratio": args.seg_filter_distance_ratio,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Wrote tracks: {out_path} ({len(tracks)} rows)")
    print(f"Wrote tracking stats: {stats_path}")


if __name__ == "__main__":
    main()
