#!/usr/bin/env python3
"""Build draft Idea 2 event GT from CVAT COCO frame-level annotations.

This script converts per-frame `state`/class labels into pairwise event intervals:
- reads a CVAT COCO JSON (e.g. instances_default.json)
- uses track ids + geometry to infer interacting pairs per frame
- merges short gaps and filters short runs
- writes Idea 2 benchmark JSON (clips/events format) for manual QA finalization
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_attr_value(attrs: Any, key: str) -> Any | None:
    if attrs is None:
        return None
    if isinstance(attrs, dict):
        if key in attrs:
            return attrs[key]
        for k, v in attrs.items():
            if str(k).strip().casefold() == key.strip().casefold():
                return v
        return None
    if isinstance(attrs, list):
        for item in attrs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name.casefold() == key.strip().casefold():
                return item.get("value")
    return None


def _extract_track_id(ann: dict[str, Any], track_id_attr: str) -> int | None:
    if isinstance(ann.get("track_id"), int):
        return int(ann["track_id"])
    val = _get_attr_value(ann.get("attributes"), track_id_attr)
    if val is not None:
        try:
            return int(str(val).strip())
        except Exception:
            pass
    gid = ann.get("group_id")
    if isinstance(gid, int):
        return int(gid)
    return None


def _state_from_ann(
    ann: dict[str, Any],
    categories_by_id: dict[int, str],
    state_attr: str,
    trophallaxis_state_value: str,
) -> str:
    raw = _get_attr_value(ann.get("attributes"), state_attr)
    if raw is not None:
        return "trophallaxis" if str(raw).strip().casefold() == trophallaxis_state_value.casefold() else "normal"
    cname = categories_by_id.get(int(ann.get("category_id", -1)), "").strip().casefold()
    return "trophallaxis" if cname == "trophallaxis" else "normal"


def _bbox_iou_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0


def _center_dist_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    acx, acy = ax + aw * 0.5, ay + ah * 0.5
    bcx, bcy = bx + bw * 0.5, by + bh * 0.5
    return float(math.hypot(acx - bcx, acy - bcy))


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _active_runs(sorted_frames: list[int]) -> list[tuple[int, int]]:
    if not sorted_frames:
        return []
    out: list[tuple[int, int]] = []
    s = p = int(sorted_frames[0])
    for f in sorted_frames[1:]:
        f = int(f)
        if f == p + 1:
            p = f
            continue
        out.append((s, p))
        s = p = f
    out.append((s, p))
    return out


def _merge_runs(runs: list[tuple[int, int]], max_gap_frames: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for st, en in runs[1:]:
        pst, pen = merged[-1]
        if st - pen - 1 <= max_gap_frames:
            merged[-1] = (pst, max(pen, en))
        else:
            merged.append((st, en))
    return merged


def _infer_sequence_name(file_name: str) -> str:
    p = Path(file_name)
    for part in reversed(p.parts[:-1]):
        if str(part).startswith("seq_"):
            return str(part)
    stem = p.stem
    m = re.match(r"^(.*?)(\d+)$", stem)
    if m and m.group(1):
        return m.group(1).rstrip("_-")
    return stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco-annotations", type=str, required=True, help="CVAT COCO annotations path.")
    p.add_argument("--out", type=str, required=True, help="Output Idea 2 benchmark JSON path.")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--state-attr", type=str, default="state")
    p.add_argument("--trophallaxis-state-value", type=str, default="trophallaxis")
    p.add_argument("--track-id-attr", type=str, default="track_id")
    p.add_argument("--max-pair-dist-px", type=float, default=90.0)
    p.add_argument("--min-pair-iou", type=float, default=0.0)
    p.add_argument("--min-active-frames", type=int, default=12)
    p.add_argument("--max-gap-frames", type=int, default=3)
    p.add_argument("--clip-allowlist", type=str, default="", help="Optional comma-separated sequence names.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    coco_path = Path(args.coco_annotations).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    coco = _read_json(coco_path)

    images = list(coco.get("images", []))
    anns = list(coco.get("annotations", []))
    cats = list(coco.get("categories", []))
    categories_by_id = {int(c.get("id", -1)): str(c.get("name", "")) for c in cats}

    img_by_id: dict[int, dict[str, Any]] = {int(im["id"]): im for im in images}
    by_seq_img: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for im in images:
        seq = _infer_sequence_name(str(im.get("file_name", "")))
        by_seq_img[seq].append(im)
    for seq in by_seq_img:
        by_seq_img[seq].sort(key=lambda im: str(im.get("file_name", "")))

    allow = {s.strip() for s in str(args.clip_allowlist).split(",") if s.strip()}
    if allow:
        by_seq_img = {k: v for k, v in by_seq_img.items() if k in allow}

    # Local frame index map per sequence (1-indexed).
    frame_idx_by_image_id: dict[int, int] = {}
    for seq, ims in by_seq_img.items():
        for idx, im in enumerate(ims, start=1):
            frame_idx_by_image_id[int(im["id"])] = int(idx)

    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in anns:
        iid = int(ann.get("image_id", -1))
        if iid in frame_idx_by_image_id:
            anns_by_image[iid].append(ann)

    clips_out: list[dict[str, Any]] = []
    for seq, ims in sorted(by_seq_img.items(), key=lambda kv: kv[0]):
        pair_frames: dict[tuple[int, int], list[int]] = defaultdict(list)
        for im in ims:
            iid = int(im["id"])
            frame = frame_idx_by_image_id[iid]
            candidates: list[dict[str, Any]] = []
            for ann in anns_by_image.get(iid, []):
                state = _state_from_ann(
                    ann,
                    categories_by_id=categories_by_id,
                    state_attr=str(args.state_attr),
                    trophallaxis_state_value=str(args.trophallaxis_state_value),
                )
                if state != "trophallaxis":
                    continue
                tid = _extract_track_id(ann, track_id_attr=str(args.track_id_attr))
                if tid is None:
                    continue
                bbox = ann.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                candidates.append({"track_id": int(tid), "bbox": [float(v) for v in bbox]})

            candidates.sort(key=lambda x: x["track_id"])
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    a = candidates[i]
                    b = candidates[j]
                    iou = _bbox_iou_xywh(a["bbox"], b["bbox"])
                    dist = _center_dist_xywh(a["bbox"], b["bbox"])
                    if iou < float(args.min_pair_iou) and dist > float(args.max_pair_dist_px):
                        continue
                    key = _pair_key(int(a["track_id"]), int(b["track_id"]))
                    pair_frames[key].append(frame)

        events: list[dict[str, Any]] = []
        for (ta, tb), frames in sorted(pair_frames.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            runs = _active_runs(sorted(set(frames)))
            runs = _merge_runs(runs, max_gap_frames=int(args.max_gap_frames))
            idx = 0
            for st, en in runs:
                dur = int(en - st + 1)
                if dur < int(args.min_active_frames):
                    continue
                idx += 1
                events.append(
                    {
                        "event_id": f"{seq}_{ta}_{tb}_{idx}",
                        "track_id_a": int(ta),
                        "track_id_b": int(tb),
                        "start_frame": int(st),
                        "end_frame": int(en),
                        "label": "trophallaxis",
                        "source": "auto_from_cvat_state",
                    }
                )

        clips_out.append(
            {
                "clip_id": seq,
                "sequence_name": seq,
                "annotation_status": "draft_auto",
                "events": events,
            }
        )

    out = {
        "version": "idea2_event_benchmark_v1",
        "description": "Auto-drafted from CVAT COCO state labels. Requires manual QA before final scoring.",
        "fps": float(args.fps),
        "source_coco_annotations": str(coco_path),
        "auto_params": {
            "state_attr": str(args.state_attr),
            "trophallaxis_state_value": str(args.trophallaxis_state_value),
            "track_id_attr": str(args.track_id_attr),
            "max_pair_dist_px": float(args.max_pair_dist_px),
            "min_pair_iou": float(args.min_pair_iou),
            "min_active_frames": int(args.min_active_frames),
            "max_gap_frames": int(args.max_gap_frames),
        },
        "notes": [
            "track_id_a and track_id_b are unordered pair IDs within sequence_name.",
            "Frame indices are 1-indexed and inclusive.",
            "Generated events are drafts and must be reviewed before final benchmarking.",
        ],
        "clips": clips_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    n_events = sum(len(c.get("events", [])) for c in clips_out)
    print(f"Wrote Idea 2 draft benchmark: {out_path}")
    print(f"Clips: {len(clips_out)} | Events: {n_events}")


if __name__ == "__main__":
    main()
