#!/usr/bin/env python3
"""Build tracker->CVAT track_id mapping from per-frame box overlaps.

This utility aligns tracker IDs (from MOT JSON) to CVAT IDs (from COCO annotations)
per sequence using greedy one-to-one IoU matching on each frame.

Optionally, it can remap predicted Idea 2 events JSON into the CVAT ID space so the
strict evaluator can be used with a canonical GT benchmark.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


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
            return None
    gid = ann.get("group_id")
    if isinstance(gid, int):
        return int(gid)
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracker-mot-json", type=str, required=True, help="MOT JSON from tracking pipeline.")
    p.add_argument("--cvat-coco-annotations", type=str, required=True, help="CVAT COCO annotations JSON.")
    p.add_argument("--out-map-json", type=str, required=True, help="Output tracker->CVAT mapping JSON path.")
    p.add_argument("--track-id-attr", type=str, default="track_id", help="CVAT attribute key for track id.")
    p.add_argument("--min-iou", type=float, default=0.3, help="Minimum IoU for frame-level pairing.")
    p.add_argument("--clip-allowlist", type=str, default="", help="Optional comma-separated sequence names.")
    p.add_argument("--pred-events-in", type=str, default="", help="Optional predicted events JSON to remap.")
    p.add_argument("--pred-events-out", type=str, default="", help="Optional output remapped events JSON path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mot_path = Path(args.tracker_mot_json).expanduser().resolve()
    coco_path = Path(args.cvat_coco_annotations).expanduser().resolve()
    out_map_path = Path(args.out_map_json).expanduser().resolve()

    allow = {s.strip() for s in str(args.clip_allowlist).split(",") if s.strip()}

    mot = json.loads(mot_path.read_text(encoding="utf-8"))
    coco = json.loads(coco_path.read_text(encoding="utf-8"))

    # tracker detections by (seq, frame)
    tracker_by_sf: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for seq_block in mot.get("sequences", []):
        seq = str(seq_block.get("sequence_name", ""))
        if allow and seq not in allow:
            continue
        for r in seq_block.get("rows", []):
            tracker_by_sf[(seq, int(r["frame"]))].append(
                {"track_id": int(r["track_id"]), "bbox": [float(v) for v in r["bbox"]]}
            )

    images = list(coco.get("images", []))
    anns = list(coco.get("annotations", []))
    img_by_id: dict[int, dict[str, Any]] = {int(im["id"]): im for im in images}
    by_seq_img: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for im in images:
        seq = _infer_sequence_name(str(im.get("file_name", "")))
        if allow and seq not in allow:
            continue
        by_seq_img[seq].append(im)
    for seq in by_seq_img:
        by_seq_img[seq].sort(key=lambda im: str(im.get("file_name", "")))

    frame_idx_by_image_id: dict[int, int] = {}
    for seq, ims in by_seq_img.items():
        for idx, im in enumerate(ims, start=1):
            frame_idx_by_image_id[int(im["id"])] = int(idx)

    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in anns:
        iid = int(ann.get("image_id", -1))
        if iid in frame_idx_by_image_id:
            anns_by_image[iid].append(ann)

    cvat_by_sf: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for seq, ims in by_seq_img.items():
        for im in ims:
            iid = int(im["id"])
            frame = frame_idx_by_image_id[iid]
            for ann in anns_by_image.get(iid, []):
                tid = _extract_track_id(ann, track_id_attr=str(args.track_id_attr))
                bbox = ann.get("bbox")
                if tid is None or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                cvat_by_sf[(seq, frame)].append({"track_id": int(tid), "bbox": [float(v) for v in bbox]})

    # Aggregate pair evidence: (seq, tracker_tid, cvat_tid) -> {count, iou_sum}
    pair_count: dict[tuple[str, int, int], int] = defaultdict(int)
    pair_iou_sum: dict[tuple[str, int, int], float] = defaultdict(float)
    seq_frames = sorted(set(tracker_by_sf.keys()) & set(cvat_by_sf.keys()))
    for sf in seq_frames:
        tr = tracker_by_sf[sf]
        cv = cvat_by_sf[sf]
        candidates: list[tuple[float, int, int]] = []
        for i, td in enumerate(tr):
            for j, cd in enumerate(cv):
                iou = _bbox_iou_xywh(td["bbox"], cd["bbox"])
                if iou >= float(args.min_iou):
                    candidates.append((iou, i, j))
        candidates.sort(key=lambda x: x[0], reverse=True)
        used_i: set[int] = set()
        used_j: set[int] = set()
        seq = sf[0]
        for iou, i, j in candidates:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            t_tid = int(tr[i]["track_id"])
            c_tid = int(cv[j]["track_id"])
            key = (seq, t_tid, c_tid)
            pair_count[key] += 1
            pair_iou_sum[key] += float(iou)

    # Resolve one-to-one mapping per sequence by strongest evidence first.
    mapping_by_seq: dict[str, dict[str, Any]] = {}
    for seq in sorted({k[0] for k in pair_count.keys()}):
        edges: list[tuple[int, float, int, int]] = []
        for (s, t_tid, c_tid), cnt in pair_count.items():
            if s != seq:
                continue
            miou = pair_iou_sum[(s, t_tid, c_tid)] / max(1, cnt)
            edges.append((cnt, miou, t_tid, c_tid))
        edges.sort(key=lambda x: (x[0], x[1]), reverse=True)
        used_t: set[int] = set()
        used_c: set[int] = set()
        chosen: list[dict[str, Any]] = []
        for cnt, miou, t_tid, c_tid in edges:
            if t_tid in used_t or c_tid in used_c:
                continue
            used_t.add(t_tid)
            used_c.add(c_tid)
            chosen.append(
                {
                    "tracker_track_id": int(t_tid),
                    "cvat_track_id": int(c_tid),
                    "support_frames": int(cnt),
                    "mean_iou": float(miou),
                }
            )
        mapping_by_seq[seq] = {
            "num_tracker_tracks_mapped": len(chosen),
            "pairs": sorted(chosen, key=lambda r: r["tracker_track_id"]),
        }

    out = {
        "format": "camponotus_track_id_mapping_v1",
        "tracker_mot_json": str(mot_path),
        "cvat_coco_annotations": str(coco_path),
        "min_iou": float(args.min_iou),
        "clip_allowlist": sorted(allow) if allow else [],
        "num_sequences_with_mapping": len(mapping_by_seq),
        "mapping_by_sequence": mapping_by_seq,
    }
    out_map_path.parent.mkdir(parents=True, exist_ok=True)
    out_map_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote tracker->CVAT mapping: {out_map_path}")

    # Optional event remap into CVAT ID space.
    if str(args.pred_events_in).strip() and str(args.pred_events_out).strip():
        pred_in_path = Path(args.pred_events_in).expanduser().resolve()
        pred_out_path = Path(args.pred_events_out).expanduser().resolve()
        pred = json.loads(pred_in_path.read_text(encoding="utf-8"))
        remapped_events: list[dict[str, Any]] = []
        dropped = 0
        for e in pred.get("events", []):
            seq = str(e.get("sequence_name", ""))
            s_map = mapping_by_seq.get(seq, {})
            id_map = {
                int(p["tracker_track_id"]): int(p["cvat_track_id"])
                for p in s_map.get("pairs", [])
            }
            ta = int(e["track_id_a"])
            tb = int(e["track_id_b"])
            if ta not in id_map or tb not in id_map:
                dropped += 1
                continue
            ra, rb = _pair_key(id_map[ta], id_map[tb])
            if ra == rb:
                dropped += 1
                continue
            out_e = dict(e)
            out_e["track_id_a"] = int(ra)
            out_e["track_id_b"] = int(rb)
            out_e["event_id"] = f"{seq}_{ra}_{rb}_{int(e.get('start_frame', 0))}"
            remapped_events.append(out_e)
        remapped = dict(pred)
        remapped["events"] = remapped_events
        remapped["id_mapping_json"] = str(out_map_path)
        remapped["remap_note"] = "track_id_a/b remapped tracker->CVAT per sequence"
        pred_out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_out_path.write_text(json.dumps(remapped, indent=2), encoding="utf-8")
        print(f"Wrote remapped events: {pred_out_path}")
        print(f"Remapped events: {len(remapped_events)} | Dropped (unmapped/collapsed): {dropped}")


if __name__ == "__main__":
    main()
