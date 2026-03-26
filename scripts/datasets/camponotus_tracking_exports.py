#!/usr/bin/env python3
"""Export helpers for Camponotus tracking outputs."""

from __future__ import annotations

import re
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _infer_sequence_key(file_name: str) -> str:
    p = Path(file_name)
    for part in reversed(p.parts[:-1]):
        if str(part).startswith("seq_"):
            return str(part)
    stem = p.stem
    m = re.match(r"^(.*?)(\d+)$", stem)
    if m and m.group(1):
        return m.group(1)
    return "__single__"


def build_mot_json_payload(
    coco_images: list[dict[str, Any]],
    preds_by_image: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build MOTChallenge-like tracking payload in JSON form."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for im in coco_images:
        seq = _infer_sequence_key(str(im["file_name"]))
        grouped.setdefault(seq, []).append(im)
    for seq in grouped:
        grouped[seq].sort(key=lambda x: str(x["file_name"]))

    sequences: list[dict[str, Any]] = []
    total_rows = 0
    for seq, items in grouped.items():
        rows: list[dict[str, Any]] = []
        for frame_idx, im in enumerate(items, start=1):
            iid = int(im["id"])
            for d in preds_by_image.get(iid, []):
                if "track_id" not in d:
                    continue
                x, y, w, h = [float(v) for v in d["bbox"]]
                rows.append(
                    {
                        "frame": int(frame_idx),
                        "track_id": int(d["track_id"]),
                        "bbox": [x, y, w, h],
                        "score": float(d.get("score", 1.0)),
                        "category_id": int(d.get("category_id", 0)),
                        "visibility": -1.0,
                        "mot_csv": (
                            f"{int(frame_idx)},{int(d['track_id'])},"
                            f"{x:.6f},{y:.6f},{w:.6f},{h:.6f},"
                            f"{float(d.get('score', 1.0)):.6f},{int(d.get('category_id', 0))},-1"
                        ),
                    }
                )
                total_rows += 1
        sequences.append(
            {
                "sequence_name": str(seq),
                "num_frames": int(len(items)),
                "rows": rows,
            }
        )
    return {
        "format": "mot_challenge_json_v1",
        "columns": [
            "frame",
            "track_id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "class",
            "vis",
        ],
        "num_sequences": int(len(sequences)),
        "num_rows": int(total_rows),
        "sequences": sequences,
    }


def write_cvat_video_xml(
    *,
    out_path: Path,
    coco_images: list[dict[str, Any]],
    preds_by_image: dict[int, list[dict[str, Any]]],
    state_normal: str = "normal",
    state_troph: str = "trophallaxis",
) -> dict[str, Any]:
    """Write CVAT Video 1.1 XML tracks with per-box `state` attribute."""
    images_sorted = sorted(coco_images, key=lambda x: str(x["file_name"]))
    frame_by_image_id: dict[int, int] = {
        int(im["id"]): idx for idx, im in enumerate(images_sorted)
    }
    n_frames_total = len(images_sorted)

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "overlap").text = "0"
    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = "ant"
    attrs = ET.SubElement(label, "attributes")
    attr = ET.SubElement(attrs, "attribute")
    ET.SubElement(attr, "name").text = "state"
    ET.SubElement(attr, "mutable").text = "true"
    ET.SubElement(attr, "input_type").text = "select"
    ET.SubElement(attr, "default_value").text = str(state_normal)
    ET.SubElement(attr, "values").text = f"{state_normal}\n{state_troph}"

    global_track_map: dict[tuple[str, int], int] = {}
    total_boxes = 0
    track_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for im in images_sorted:
        iid = int(im["id"])
        seq_name = _infer_sequence_key(str(im["file_name"]))
        frame_idx = int(frame_by_image_id[iid])
        for d in preds_by_image.get(iid, []):
            if "track_id" not in d:
                continue
            raw_tid = int(d["track_id"])
            key = (seq_name, raw_tid)
            x, y, w, h = [float(v) for v in d["bbox"]]
            track_rows.setdefault(key, []).append(
                {
                    "frame": frame_idx,
                    "xtl": x,
                    "ytl": y,
                    "xbr": x + max(0.0, w),
                    "ybr": y + max(0.0, h),
                    "state": state_troph if int(d.get("category_id", 0)) == 1 else state_normal,
                }
            )

    track_elements: dict[int, ET.Element] = {}
    for tid_num, key in enumerate(sorted(track_rows.keys(), key=lambda k: (k[0], k[1]))):
        global_track_map[key] = int(tid_num)
        track_elements[int(tid_num)] = ET.SubElement(
            root, "track", {"id": str(int(tid_num)), "label": "ant", "source": "auto"}
        )

    for key, rows in track_rows.items():
        track_el = track_elements[global_track_map[key]]
        rows_sorted = sorted(rows, key=lambda r: int(r["frame"]))
        for idx, r in enumerate(rows_sorted):
            frame = int(r["frame"])
            box = ET.SubElement(
                track_el,
                "box",
                {
                    "frame": str(frame),
                    "outside": "0",
                    "occluded": "0",
                    "keyframe": "1",
                    "z_order": "0",
                    "xtl": f"{float(r['xtl']):.6f}",
                    "ytl": f"{float(r['ytl']):.6f}",
                    "xbr": f"{float(r['xbr']):.6f}",
                    "ybr": f"{float(r['ybr']):.6f}",
                },
            )
            attr_el = ET.SubElement(box, "attribute", {"name": "state"})
            attr_el.text = str(r["state"])
            total_boxes += 1

            next_frame = int(rows_sorted[idx + 1]["frame"]) if idx + 1 < len(rows_sorted) else None
            gap_start = frame + 1
            should_close_gap = next_frame is None or next_frame > gap_start
            if should_close_gap and gap_start < n_frames_total:
                ET.SubElement(
                    track_el,
                    "box",
                    {
                        "frame": str(gap_start),
                        "outside": "1",
                        "occluded": "0",
                        "keyframe": "1",
                        "z_order": "0",
                        "xtl": f"{float(r['xtl']):.6f}",
                        "ytl": f"{float(r['ytl']):.6f}",
                        "xbr": f"{float(r['xbr']):.6f}",
                        "ybr": f"{float(r['ybr']):.6f}",
                    },
                )

    xml_bytes = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_bytes).toprettyxml(indent="  ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(pretty_xml, encoding="utf-8")
    return {
        "num_tracks": int(len(global_track_map)),
        "num_boxes": int(total_boxes),
        "num_sequences": int(len({_infer_sequence_key(str(im["file_name"])) for im in coco_images})),
        "num_frames_total": int(n_frames_total),
    }
