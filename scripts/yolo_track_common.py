#!/usr/bin/env python3
"""Shared YOLO tracking helpers for video and frame-sequence pipelines."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import tempfile

import yaml


def build_tracker_config(
    *,
    tracker: str,
    track_thresh: float,
    match_thresh: float,
    track_buffer: int,
    with_reid: bool = False,
    proximity_thresh: float = 0.5,
    appearance_thresh: float = 0.25,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "tracker_type": str(tracker),
        "track_high_thresh": float(track_thresh),
        "track_low_thresh": max(0.01, float(track_thresh) * 0.5),
        "new_track_thresh": max(0.01, float(track_thresh) * 0.8),
        "track_buffer": int(track_buffer),
        "match_thresh": float(match_thresh),
        "fuse_score": True,
    }
    if str(tracker) == "botsort":
        cfg.update(
            {
                "gmc_method": "sparseOptFlow",
                "proximity_thresh": float(proximity_thresh),
                "appearance_thresh": float(appearance_thresh),
                "with_reid": bool(with_reid),
                "model": "auto" if bool(with_reid) else "",
            }
        )
    return cfg


@contextmanager
def temporary_tracker_yaml(
    tracker_cfg: dict[str, Any], *, suffix: str = "_tracker.yaml"
) -> Iterator[Path]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tf:
        yaml.safe_dump(tracker_cfg, tf, sort_keys=False)
        p = Path(tf.name)
    try:
        yield p
    finally:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def iter_tracked_detections(
    model: Any,
    *,
    source: str,
    conf: float,
    tracker_cfg_path: Path,
    imgsz: int | None = None,
    device: str | None = None,
    persist: bool = True,
    stream: bool = True,
    verbose: bool = False,
) -> Iterator[tuple[Any, list[dict[str, Any]]]]:
    kw: dict[str, Any] = {
        "source": str(source),
        "conf": float(conf),
        "tracker": str(tracker_cfg_path),
        "persist": bool(persist),
        "stream": bool(stream),
        "verbose": bool(verbose),
    }
    if imgsz is not None:
        kw["imgsz"] = int(imgsz)
    if device is not None:
        kw["device"] = str(device)

    results = model.track(**kw)
    if results is None:
        return
    for r in results:
        boxes = getattr(r, "boxes", None)
        dets: list[dict[str, Any]] = []
        if boxes is not None and len(boxes) > 0 and getattr(boxes, "id", None) is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            clss = boxes.cls.cpu().numpy() if boxes.cls is not None else None
            tids = boxes.id.cpu().numpy()
            for i in range(len(xyxy)):
                dets.append(
                    {
                        "xyxy": [float(v) for v in xyxy[i].tolist()],
                        "track_id": int(tids[i]),
                        "class_id": int(clss[i]) if clss is not None and i < len(clss) else -1,
                        "score": float(confs[i]) if confs is not None and i < len(confs) else 0.0,
                    }
                )
        yield r, dets
