#!/usr/bin/env python3
"""Shared temporal utilities for EXP-A006 (tracking + smoothing)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


def load_coco_images(gt_path: Path) -> list[dict[str, Any]]:
    coco = json.loads(gt_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for im in coco.get("images", []):
        if im.get("id") is None or im.get("file_name") is None:
            continue
        out.append({"id": int(im["id"]), "file_name": str(im["file_name"])})
    return out


def infer_sequence_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    m = re.match(r"^(.*?)(\d+)$", stem)
    if m and m.group(1):
        return m.group(1)
    return "__single__"


def load_sequence_map_from_manifest(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    seq = raw.get("sequence_map")
    if isinstance(seq, dict):
        return {str(k): str(v) for k, v in seq.items()}
    return {}


def group_frames(
    images: list[dict[str, Any]],
    sequence_map: dict[str, str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[int, tuple[str, int]], dict[tuple[str, int], int]]:
    seq_map = sequence_map or {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for im in images:
        fn = Path(str(im["file_name"])).name
        seq = seq_map.get(fn) or infer_sequence_from_filename(fn)
        grouped.setdefault(seq, []).append(im)
    for seq in grouped:
        grouped[seq].sort(key=lambda x: Path(str(x["file_name"])).name)

    img_to_pos: dict[int, tuple[str, int]] = {}
    pos_to_img: dict[tuple[str, int], int] = {}
    for seq, items in grouped.items():
        for idx, im in enumerate(items):
            iid = int(im["id"])
            img_to_pos[iid] = (seq, idx)
            pos_to_img[(seq, idx)] = iid
    return grouped, img_to_pos, pos_to_img


def load_predictions(pred_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(pred_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("annotations"), list):
        return list(raw["annotations"])
    raise ValueError("Predictions must be COCO list or {'annotations': [...]}")


def _xywh_to_xyxy(b: list[float]) -> list[float]:
    x, y, w, h = map(float, b)
    return [x, y, x + max(0.0, w), y + max(0.0, h)]


def _xyxy_to_xywh(b: list[float]) -> list[float]:
    x1, y1, x2, y2 = map(float, b)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def run_bytetrack_on_predictions(
    preds: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    img_to_pos: dict[int, tuple[str, int]],
    track_thresh: float = 0.25,
    match_thresh: float = 0.8,
    track_buffer: int = 30,
    seg_filter_distance_abs: float | None = None,
    seg_filter_distance_ratio: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import supervision as sv
    except ImportError as exc:
        raise RuntimeError(
            "supervision is required for EXP-A006 tracking. Install with: pip install supervision"
        ) from exc

    by_img: dict[int, list[dict[str, Any]]] = {}
    for d in preds:
        iid = d.get("image_id")
        if iid is None:
            continue
        by_img.setdefault(int(iid), []).append(d)

    def _extract_masks(frame_dets: list[dict[str, Any]]) -> np.ndarray | None:
        masks: list[np.ndarray] = []
        for det in frame_dets:
            m = det.get("mask")
            if m is None:
                return None
            arr = np.asarray(m, dtype=bool)
            if arr.ndim != 2:
                return None
            masks.append(arr)
        if not masks:
            return None
        shape0 = masks[0].shape
        if any(msk.shape != shape0 for msk in masks):
            return None
        return np.stack(masks, axis=0)

    t0 = time.perf_counter()
    tracks: list[dict[str, Any]] = []
    for seq, items in grouped.items():
        tracker = sv.ByteTrack(
            track_activation_threshold=float(track_thresh),
            minimum_matching_threshold=float(match_thresh),
            lost_track_buffer=int(track_buffer),
        )
        for frame_idx, im in enumerate(items):
            iid = int(im["id"])
            frame_dets = by_img.get(iid, [])
            if frame_dets:
                xyxy = np.asarray([_xywh_to_xyxy(d["bbox"]) for d in frame_dets], dtype=np.float32)
                conf = np.asarray(
                    [float(d.get("score", 1.0)) for d in frame_dets],
                    dtype=np.float32,
                )
                cls = np.asarray(
                    [int(d.get("category_id", 0)) for d in frame_dets],
                    dtype=np.int32,
                )
            else:
                xyxy = np.zeros((0, 4), dtype=np.float32)
                conf = np.zeros((0,), dtype=np.float32)
                cls = np.zeros((0,), dtype=np.int32)
            dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
            masks = _extract_masks(frame_dets)
            if masks is not None and masks.shape[0] == xyxy.shape[0]:
                dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls, mask=masks)
                filter_kwargs: dict[str, float] = {}
                if seg_filter_distance_abs is not None:
                    filter_kwargs["distance_threshold"] = float(seg_filter_distance_abs)
                if seg_filter_distance_ratio is not None:
                    filter_kwargs["distance_ratio_threshold"] = float(seg_filter_distance_ratio)
                if filter_kwargs:
                    dets = sv.filter_segments_by_distance(dets, **filter_kwargs)
            out = tracker.update_with_detections(dets)
            out_xyxy = np.asarray(getattr(out, "xyxy", np.zeros((0, 4))))
            out_conf = np.asarray(getattr(out, "confidence", np.zeros((0,))))
            out_cls = np.asarray(getattr(out, "class_id", np.zeros((0,))))
            out_tid = np.asarray(getattr(out, "tracker_id", np.zeros((0,))))
            for i in range(out_xyxy.shape[0]):
                tid = out_tid[i] if i < len(out_tid) else -1
                if tid is None or int(tid) < 0:
                    continue
                bbox = _xyxy_to_xywh(out_xyxy[i].tolist())
                tracks.append(
                    {
                        "sequence_id": seq,
                        "frame_index": frame_idx,
                        "image_id": iid,
                        "file_name": str(im["file_name"]),
                        "track_id": int(tid),
                        "category_id": int(out_cls[i]) if i < len(out_cls) else 0,
                        "bbox": bbox,
                        "score": float(out_conf[i]) if i < len(out_conf) else 1.0,
                    }
                )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    uniq_tracks = {(t["sequence_id"], t["track_id"]) for t in tracks}
    stats = {
        "n_tracks_total": len(uniq_tracks),
        "n_detections_tracked": len(tracks),
        "elapsed_ms": elapsed_ms,
        "segmentation_filter_enabled": bool(
            seg_filter_distance_abs is not None or seg_filter_distance_ratio is not None
        ),
    }
    return tracks, stats


def smooth_tracks(
    tracks: list[dict[str, Any]],
    pos_to_img: dict[tuple[str, int], int],
    min_track_len: int = 3,
    fill_gap_max: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t0 = time.perf_counter()
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for t in tracks:
        key = (str(t["sequence_id"]), int(t["track_id"]))
        grouped.setdefault(key, []).append(dict(t))
    for key in grouped:
        grouped[key].sort(key=lambda x: int(x["frame_index"]))

    removed = 0
    smoothed: list[dict[str, Any]] = []
    lengths: list[int] = []

    for (seq, tid), items in grouped.items():
        if len(items) < int(min_track_len):
            removed += 1
            continue
        lengths.append(len(items))
        score_avg = float(np.mean([float(x.get("score", 1.0)) for x in items]))
        full: list[dict[str, Any]] = []
        for i, cur in enumerate(items):
            cur = dict(cur)
            cur["score"] = score_avg
            full.append(cur)
            if i + 1 >= len(items):
                continue
            nxt = items[i + 1]
            gap = int(nxt["frame_index"]) - int(cur["frame_index"]) - 1
            if gap <= 0 or gap > int(fill_gap_max):
                continue
            cbox = np.asarray(cur["bbox"], dtype=np.float64)
            nbox = np.asarray(nxt["bbox"], dtype=np.float64)
            for g in range(1, gap + 1):
                alpha = g / float(gap + 1)
                ibox = (1.0 - alpha) * cbox + alpha * nbox
                fidx = int(cur["frame_index"]) + g
                iid = pos_to_img.get((seq, fidx))
                if iid is None:
                    continue
                full.append(
                    {
                        "sequence_id": seq,
                        "frame_index": fidx,
                        "image_id": int(iid),
                        "file_name": "",
                        "track_id": int(tid),
                        "category_id": int(cur.get("category_id", 0)),
                        "bbox": [float(v) for v in ibox.tolist()],
                        "score": score_avg,
                    }
                )
        smoothed.extend(full)

    smoothed.sort(key=lambda x: (str(x["sequence_id"]), int(x["frame_index"]), int(x["track_id"])))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    stats = {
        "removed_tracks_count": int(removed),
        "n_tracks_kept": int(len(lengths)),
        "avg_track_length": avg_len,
        "n_smoothed_detections": len(smoothed),
        "elapsed_ms": elapsed_ms,
    }
    return smoothed, stats


def tracks_to_coco_predictions(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tracks:
        out.append(
            {
                "image_id": int(t["image_id"]),
                "category_id": int(t.get("category_id", 0)),
                "bbox": [float(v) for v in t["bbox"]],
                "score": float(t.get("score", 1.0)),
            }
        )
    return out
