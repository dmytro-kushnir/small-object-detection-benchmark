#!/usr/bin/env python3
"""Generate model prelabels (COCO detections) for Camponotus annotation bootstrap."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from camponotus_common import CAMPO_CLASSES, build_categories, write_json
# Allow direct script execution (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_track_common import (
    build_tracker_config,
    iter_tracked_detections,
    temporary_tracker_yaml,
)
from datasets.camponotus_tracking_exports import (
    build_mot_json_payload,
    write_cvat_video_xml,
)


def _collect_images(images_root: Path, max_images: int | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = sorted(p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if max_images is not None:
        imgs = imgs[: max(0, max_images)]
    return imgs


def _extract_video_frames_to_png(video_path: Path, max_frames: int | None = None) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="campo_video_frames_"))
    pattern = str((tmp_dir / "frame_%08d.png").resolve())
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vsync",
        "0",
    ]
    if max_frames is not None:
        cmd.extend(["-frames:v", str(int(max_frames))])
    cmd.append(pattern)
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return tmp_dir


def _infer_with_yolo(
    model: Any, image: Any, conf: float, imgsz: int | None = None
) -> list[dict[str, Any]]:
    kw: dict[str, Any] = {"conf": conf, "verbose": False}
    if imgsz is not None:
        kw["imgsz"] = int(imgsz)
    results = model.predict(image, **kw)
    out: list[dict[str, Any]] = []
    if not results:
        return out
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy()
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
        out.append(
            {
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(confs[i]),
                "category_id": int(clss[i]) if int(clss[i]) in (0, 1) else 0,
            }
        )
    return out


def _infer_with_rfdetr(model: Any, image: Any, conf: float) -> list[dict[str, Any]]:
    raw = model.predict(image, threshold=float(conf))
    det = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
    if det is None:
        return []
    xyxy = getattr(det, "xyxy", None)
    if xyxy is None:
        return []
    confs = getattr(det, "confidence", None)
    clss = getattr(det, "class_id", None)
    out: list[dict[str, Any]] = []
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        score = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        cid = int(clss[i]) if clss is not None and i < len(clss) else 0
        out.append(
            {
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": score,
                "category_id": cid if cid in (0, 1) else 0,
            }
        )
    return out


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


def _xywh_to_xyxy(b: list[float]) -> list[float]:
    x, y, w, h = map(float, b)
    return [x, y, x + max(0.0, w), y + max(0.0, h)]


def _xyxy_to_xywh(b: list[float]) -> list[float]:
    x1, y1, x2, y2 = map(float, b)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _bbox_iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _state_priority_soft_relabel(
    dets: list[dict[str, Any]],
    *,
    iou_thresh: float,
    score_gap_max: float,
) -> tuple[list[dict[str, Any]], int]:
    troph = [d for d in dets if int(d.get("category_id", 0)) == 1]
    if not troph:
        return dets, 0
    out: list[dict[str, Any]] = []
    relabeled = 0
    for d in dets:
        if int(d.get("category_id", 0)) != 0:
            out.append(d)
            continue
        n_score = float(d.get("score", 0.0))
        should_flip = False
        for t in troph:
            iou = _bbox_iou_xywh(
                [float(x) for x in d["bbox"]],
                [float(x) for x in t["bbox"]],
            )
            if iou < float(iou_thresh):
                continue
            score_gap = float(t.get("score", 0.0)) - n_score
            if score_gap >= 0.0 and score_gap <= float(score_gap_max):
                should_flip = True
                break
        if should_flip:
            nd = dict(d)
            nd["category_id"] = 1
            out.append(nd)
            relabeled += 1
        else:
            out.append(d)
    return out, relabeled


def _build_mot_json(
    coco_images: list[dict[str, Any]],
    preds_by_image: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    return build_mot_json_payload(coco_images=coco_images, preds_by_image=preds_by_image)


def _write_cvat_video_xml(
    *,
    out_path: Path,
    coco_images: list[dict[str, Any]],
    preds_by_image: dict[int, list[dict[str, Any]]],
    state_normal: str = "normal",
    state_troph: str = "trophallaxis",
) -> dict[str, Any]:
    return write_cvat_video_xml(
        out_path=out_path,
        coco_images=coco_images,
        preds_by_image=preds_by_image,
        state_normal=state_normal,
        state_troph=state_troph,
    )


def _apply_tracking_bytetrack(
    coco_images: list[dict[str, Any]],
    preds_by_image: dict[int, list[dict[str, Any]]],
    track_thresh: float,
    match_thresh: float,
    track_buffer: int,
    min_track_len: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    try:
        import supervision as sv
    except ImportError as exc:
        raise RuntimeError(
            "Tracking requires supervision. Install with: pip install supervision"
        ) from exc

    grouped: dict[str, list[dict[str, Any]]] = {}
    for im in coco_images:
        seq = _infer_sequence_key(str(im["file_name"]))
        grouped.setdefault(seq, []).append(im)
    for seq in grouped:
        grouped[seq].sort(key=lambda x: str(x["file_name"]))

    out_by_image: dict[int, list[dict[str, Any]]] = {}
    img_to_seq: dict[int, str] = {}
    track_lengths: dict[tuple[str, int], int] = {}
    for seq, items in grouped.items():
        tracker = sv.ByteTrack(
            track_activation_threshold=float(track_thresh),
            minimum_matching_threshold=float(match_thresh),
            lost_track_buffer=int(track_buffer),
        )
        for im in items:
            iid = int(im["id"])
            img_to_seq[iid] = seq
            frame_dets = preds_by_image.get(iid, [])
            if frame_dets:
                xyxy = np.asarray(
                    [_xywh_to_xyxy(d["bbox"]) for d in frame_dets],
                    dtype=np.float32,
                )
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
            trk = tracker.update_with_detections(dets)
            trk_xyxy = np.asarray(getattr(trk, "xyxy", np.zeros((0, 4))))
            trk_conf = np.asarray(getattr(trk, "confidence", np.zeros((0,))))
            trk_cls = np.asarray(getattr(trk, "class_id", np.zeros((0,))))
            trk_tid = np.asarray(getattr(trk, "tracker_id", np.zeros((0,))))
            frame_out: list[dict[str, Any]] = []
            for j in range(trk_xyxy.shape[0]):
                tid = trk_tid[j] if j < len(trk_tid) else -1
                if tid is None or int(tid) < 0:
                    continue
                track_lengths[(seq, int(tid))] = track_lengths.get((seq, int(tid)), 0) + 1
                frame_out.append(
                    {
                        "bbox": _xyxy_to_xywh(trk_xyxy[j].tolist()),
                        "score": float(trk_conf[j]) if j < len(trk_conf) else 1.0,
                        "category_id": int(trk_cls[j]) if j < len(trk_cls) else 0,
                        "track_id": int(tid),
                    }
                )
            out_by_image[iid] = frame_out

    if int(min_track_len) > 1:
        for iid in list(out_by_image.keys()):
            seq = img_to_seq.get(iid, "__single__")
            out_by_image[iid] = [
                d
                for d in out_by_image[iid]
                if track_lengths.get((seq, int(d["track_id"])), 0) >= int(min_track_len)
            ]

    kept_tracks = {
        (seq, tid)
        for (seq, tid), n in track_lengths.items()
        if n >= int(min_track_len)
    }
    stats = {
        "enabled": True,
        "track_thresh": float(track_thresh),
        "match_thresh": float(match_thresh),
        "track_buffer": int(track_buffer),
        "min_track_len": int(min_track_len),
        "n_sequences": len(grouped),
        "n_tracks_total": len(track_lengths),
        "n_tracks_kept": len(kept_tracks),
        "n_detections_after_tracking": int(sum(len(v) for v in out_by_image.values())),
    }
    return out_by_image, stats


def _apply_tracking_botsort_yolo(
    *,
    model: Any,
    coco_images: list[dict[str, Any]],
    images_root: Path,
    conf: float,
    imgsz: int | None,
    track_thresh: float,
    match_thresh: float,
    track_buffer: int,
    min_track_len: int,
    with_reid: bool,
    proximity_thresh: float,
    appearance_thresh: float,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for im in coco_images:
        seq = _infer_sequence_key(str(im["file_name"]))
        grouped.setdefault(seq, []).append(im)
    for seq in grouped:
        grouped[seq].sort(key=lambda x: str(x["file_name"]))

    tracker_cfg = build_tracker_config(
        tracker="botsort",
        track_thresh=float(track_thresh),
        match_thresh=float(match_thresh),
        track_buffer=int(track_buffer),
        with_reid=bool(with_reid),
        proximity_thresh=float(proximity_thresh),
        appearance_thresh=float(appearance_thresh),
    )

    out_by_image: dict[int, list[dict[str, Any]]] = {}
    img_to_seq: dict[int, str] = {}
    track_lengths: dict[tuple[str, int], int] = {}
    with temporary_tracker_yaml(tracker_cfg, suffix="_botsort.yaml") as tracker_cfg_path:
        for seq, items in grouped.items():
            src = [str((images_root / str(it["file_name"])).resolve()) for it in items]
            if not src:
                for im in items:
                    out_by_image[int(im["id"])] = []
                continue
            seq_src_dir = Path(tempfile.mkdtemp(prefix="campo_botsort_seq_"))
            for idx, src_path in enumerate(src):
                p = Path(src_path)
                ext = p.suffix.lower() or ".jpg"
                link_name = seq_src_dir / f"frame_{idx:08d}{ext}"
                try:
                    link_name.symlink_to(p)
                except Exception:
                    shutil.copy2(str(p), str(link_name))
            kw: dict[str, Any] = {
                "source": str(seq_src_dir),
            }
            try:
                results_iter = iter_tracked_detections(
                    model,
                    source=kw["source"],
                    conf=float(conf),
                    tracker_cfg_path=tracker_cfg_path,
                    imgsz=int(imgsz) if imgsz is not None else None,
                    persist=True,
                    stream=True,
                    verbose=False,
                )
                for idx, im in enumerate(items):
                    iid = int(im["id"])
                    img_to_seq[iid] = seq
                    _, dets = next(results_iter, (None, []))
                    frame_out: list[dict[str, Any]] = []
                    for d in dets:
                        tid = int(d.get("track_id", -1))
                        if tid < 0:
                            continue
                        track_lengths[(seq, tid)] = track_lengths.get((seq, tid), 0) + 1
                        xyxy = [float(v) for v in d.get("xyxy", [0.0, 0.0, 0.0, 0.0])]
                        cid = int(d.get("class_id", 0))
                        frame_out.append(
                            {
                                "bbox": _xyxy_to_xywh(xyxy),
                                "score": float(d.get("score", 0.0)),
                                "category_id": cid if cid in (0, 1) else 0,
                                "track_id": tid,
                            }
                        )
                    out_by_image[iid] = frame_out
            finally:
                try:
                    shutil.rmtree(seq_src_dir, ignore_errors=True)
                except Exception:
                    pass

    if int(min_track_len) > 1:
        for iid in list(out_by_image.keys()):
            seq = img_to_seq.get(iid, "__single__")
            out_by_image[iid] = [
                d
                for d in out_by_image[iid]
                if track_lengths.get((seq, int(d["track_id"])), 0) >= int(min_track_len)
            ]

    kept_tracks = {(seq, tid) for (seq, tid), n in track_lengths.items() if n >= int(min_track_len)}
    stats = {
        "enabled": True,
        "tracker": "botsort",
        "with_reid": bool(with_reid),
        "track_thresh": float(track_thresh),
        "match_thresh": float(match_thresh),
        "track_buffer": int(track_buffer),
        "min_track_len": int(min_track_len),
        "proximity_thresh": float(proximity_thresh),
        "appearance_thresh": float(appearance_thresh),
        "n_sequences": len(grouped),
        "n_tracks_total": len(track_lengths),
        "n_tracks_kept": len(kept_tracks),
        "n_detections_after_tracking": int(sum(len(v) for v in out_by_image.values())),
    }
    return out_by_image, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-root", type=str, default="datasets/camponotus_raw")
    p.add_argument(
        "--source-video",
        type=str,
        default="",
        help="Optional video source; extracted to temporary PNG frames for prelabeling.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="datasets/camponotus_processed/prelabels/camponotus_prelabels_coco.json",
    )
    p.add_argument("--backend", choices=("auto", "yolo", "rfdetr"), default="auto")
    p.add_argument("--yolo-weights", type=str, default="")
    p.add_argument("--rfdetr-weights", type=str, default="")
    p.add_argument("--rfdetr-model-class", type=str, default="RFDETRSmall")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--yolo-imgsz",
        type=int,
        default=None,
        help="Optional Ultralytics predict imgsz (e.g. 768 to match ants_expA002b_imgsz768 training).",
    )
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument(
        "--with-tracking",
        action="store_true",
        help="Apply tracking over sequential frames and emit stable track_id on annotations.",
    )
    p.add_argument(
        "--tracker",
        choices=("bytetrack", "botsort"),
        default="bytetrack",
        help="Tracking backend when --with-tracking is enabled.",
    )
    p.add_argument(
        "--track-thresh",
        type=float,
        default=0.25,
        help="ByteTrack activation threshold for detections.",
    )
    p.add_argument(
        "--match-thresh",
        type=float,
        default=0.8,
        help="ByteTrack IoU matching threshold.",
    )
    p.add_argument(
        "--track-buffer",
        type=int,
        default=30,
        help="ByteTrack lost-track buffer in frames.",
    )
    p.add_argument(
        "--min-track-len",
        type=int,
        default=2,
        help="Drop tracked boxes from very short tracks (< this length).",
    )
    p.add_argument(
        "--botsort-with-reid",
        action="store_true",
        help="Enable appearance ReID for BoT-SORT (YOLO backend only).",
    )
    p.add_argument(
        "--botsort-proximity-thresh",
        type=float,
        default=0.5,
        help="BoT-SORT proximity threshold.",
    )
    p.add_argument(
        "--botsort-appearance-thresh",
        type=float,
        default=0.25,
        help="BoT-SORT appearance threshold (used when ReID is enabled).",
    )
    p.add_argument(
        "--cvat-coco-categories",
        action="store_true",
        help=(
            "Use COCO category ids 1..N (not 0..N-1). Required for CVAT COCO import; "
            "see scripts/datasets/coco_shift_category_ids_for_cvat.py for existing JSON."
        ),
    )
    p.add_argument(
        "--state-priority-soft",
        action="store_true",
        help=(
            "Soft state preference: relabel normal->trophallaxis when overlap is high "
            "and score gap is small (keeps all detections)."
        ),
    )
    p.add_argument(
        "--state-priority-iou-thresh",
        type=float,
        default=0.7,
        help="IoU threshold for --state-priority-soft.",
    )
    p.add_argument(
        "--state-priority-score-gap-max",
        type=float,
        default=0.12,
        help="Max (trophallaxis_score - normal_score) to allow soft relabel.",
    )
    p.add_argument(
        "--mot-out-json",
        type=str,
        default="",
        help=(
            "Optional MOT-style JSON sidecar output. "
            "Works best with --with-tracking to include track_id."
        ),
    )
    p.add_argument(
        "--cvat-video-xml-out",
        type=str,
        default="",
        help=(
            "Optional CVAT Video 1.1 XML export with track interpolation "
            "and per-box `state` attribute."
        ),
    )
    args = p.parse_args()

    source_video = Path(args.source_video).expanduser().resolve() if str(args.source_video).strip() else None
    images_root = Path(args.images_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    mot_out_json = Path(args.mot_out_json).expanduser().resolve() if str(args.mot_out_json).strip() else None
    cvat_video_xml_out = (
        Path(args.cvat_video_xml_out).expanduser().resolve()
        if str(args.cvat_video_xml_out).strip()
        else None
    )
    if source_video is not None:
        if not source_video.is_file():
            print(f"source video not found: {source_video}", file=sys.stderr)
            sys.exit(1)
    elif not images_root.is_dir():
        print(f"images root not found: {images_root}", file=sys.stderr)
        sys.exit(1)

    yolo_w = Path(args.yolo_weights).expanduser().resolve() if args.yolo_weights else None
    rfd_w = Path(args.rfdetr_weights).expanduser().resolve() if args.rfdetr_weights else None
    backend = args.backend
    if backend == "auto":
        if rfd_w is not None and rfd_w.is_file():
            backend = "rfdetr"
        elif yolo_w is not None and yolo_w.is_file():
            backend = "yolo"
        else:
            print(
                "auto backend requires at least one valid --yolo-weights or --rfdetr-weights path",
                file=sys.stderr,
            )
            sys.exit(1)

    if backend == "yolo" and (yolo_w is None or not yolo_w.is_file()):
        print("YOLO backend selected but --yolo-weights is missing/invalid", file=sys.stderr)
        sys.exit(1)
    if backend == "rfdetr" and (rfd_w is None or not rfd_w.is_file()):
        print("RF-DETR backend selected but --rfdetr-weights is missing/invalid", file=sys.stderr)
        sys.exit(1)
    if args.with_tracking and args.tracker == "botsort" and backend != "yolo":
        print("BoT-SORT is currently supported only with YOLO backend in this script", file=sys.stderr)
        sys.exit(1)

    temp_images_root: Path | None = None
    try:
        if source_video is not None:
            temp_images_root = _extract_video_frames_to_png(source_video, args.max_images)
            images_root = temp_images_root
        images = _collect_images(images_root, args.max_images)
        if not images:
            print("No images found for prelabel generation", file=sys.stderr)
            sys.exit(1)

        infer_model: Any
        if backend == "yolo":
            from ultralytics import YOLO

            infer_model = YOLO(str(yolo_w))
        else:
            import importlib

            mc = str(args.rfdetr_model_class)
            rfdetr = importlib.import_module("rfdetr")
            if not hasattr(rfdetr, mc):
                raise ValueError(f"Unknown RF-DETR model class: {mc}")
            Model = getattr(rfdetr, mc)
            infer_model = Model(pretrain_weights=str(rfd_w))

        coco_images: list[dict[str, Any]] = []
        coco_anns: list[dict[str, Any]] = []
        ann_id = 1
        cat_offset = 1 if args.cvat_coco_categories else 0
        preds_by_image: dict[int, list[dict[str, Any]]] = {}
        for i, img_path in enumerate(images, start=1):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            coco_images.append(
                {
                    "id": i,
                    "file_name": str(img_path.resolve().relative_to(images_root)),
                    "width": int(w),
                    "height": int(h),
                }
            )
            if args.with_tracking and backend == "yolo" and args.tracker == "botsort":
                preds_by_image[i] = []
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if backend == "yolo":
                    imgsz = int(args.yolo_imgsz) if args.yolo_imgsz is not None else None
                    preds = _infer_with_yolo(infer_model, rgb, conf=float(args.conf), imgsz=imgsz)
                else:
                    preds = _infer_with_rfdetr(infer_model, rgb, conf=float(args.conf))
                preds_by_image[i] = preds

        tracking_stats: dict[str, Any] | None = None
        if args.with_tracking:
            if args.tracker == "botsort":
                imgsz = int(args.yolo_imgsz) if args.yolo_imgsz is not None else None
                preds_by_image, tracking_stats = _apply_tracking_botsort_yolo(
                    model=infer_model,
                    coco_images=coco_images,
                    images_root=images_root,
                    conf=float(args.conf),
                    imgsz=imgsz,
                    track_thresh=float(args.track_thresh),
                    match_thresh=float(args.match_thresh),
                    track_buffer=int(args.track_buffer),
                    min_track_len=int(args.min_track_len),
                    with_reid=bool(args.botsort_with_reid),
                    proximity_thresh=float(args.botsort_proximity_thresh),
                    appearance_thresh=float(args.botsort_appearance_thresh),
                )
            else:
                preds_by_image, tracking_stats = _apply_tracking_bytetrack(
                    coco_images=coco_images,
                    preds_by_image=preds_by_image,
                    track_thresh=float(args.track_thresh),
                    match_thresh=float(args.match_thresh),
                    track_buffer=int(args.track_buffer),
                    min_track_len=int(args.min_track_len),
                )

        soft_relabel_count = 0
        if args.state_priority_soft:
            for iid in list(preds_by_image.keys()):
                updated, n = _state_priority_soft_relabel(
                    preds_by_image.get(iid, []),
                    iou_thresh=float(args.state_priority_iou_thresh),
                    score_gap_max=float(args.state_priority_score_gap_max),
                )
                preds_by_image[iid] = updated
                soft_relabel_count += int(n)

        for im in coco_images:
            image_id = int(im["id"])
            for pdet in preds_by_image.get(image_id, []):
                ann = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(pdet["category_id"]) + cat_offset,
                    "bbox": [float(x) for x in pdet["bbox"]],
                    "score": float(pdet["score"]),
                    "area": float(pdet["bbox"][2] * pdet["bbox"][3]),
                    "iscrowd": 0,
                }
                if "track_id" in pdet:
                    ann["track_id"] = int(pdet["track_id"])
                coco_anns.append(
                    ann
                )
                ann_id += 1

        if args.cvat_coco_categories:
            coco_categories = []
            for c in build_categories():
                cid = int(c["id"])
                coco_categories.append(
                    {"id": cid + 1, "name": c["name"], "supercategory": c.get("supercategory", "object")}
                )
        else:
            coco_categories = [{"id": 0, "name": "ant"}, {"id": 1, "name": "trophallaxis"}]

        payload = {
            "info": {
                "description": "Camponotus prelabels for CVAT correction",
                "backend": backend,
                "classes": CAMPO_CLASSES,
                "cvat_coco_categories": bool(args.cvat_coco_categories),
            },
            "images": coco_images,
            "annotations": coco_anns,
            "categories": coco_categories,
        }
        write_json(out_path, payload)
        if mot_out_json is not None:
            mot_payload = _build_mot_json(coco_images=coco_images, preds_by_image=preds_by_image)
            write_json(mot_out_json, mot_payload)
        cvat_video_stats: dict[str, Any] | None = None
        if cvat_video_xml_out is not None:
            cvat_video_stats = _write_cvat_video_xml(
                out_path=cvat_video_xml_out,
                coco_images=coco_images,
                preds_by_image=preds_by_image,
            )

        manifest = {
            "backend": backend,
            "images_root": str(images_root),
            "source_video": str(source_video) if source_video is not None else None,
            "output_json": str(out_path),
            "mot_output_json": str(mot_out_json) if mot_out_json is not None else None,
            "cvat_video_xml_out": (
                str(cvat_video_xml_out) if cvat_video_xml_out is not None else None
            ),
            "images_processed": len(coco_images),
            "annotations_generated": len(coco_anns),
            "conf": float(args.conf),
            "yolo_imgsz": int(args.yolo_imgsz) if args.yolo_imgsz is not None else None,
            "weights": {
                "yolo": str(yolo_w) if yolo_w else None,
                "rfdetr": str(rfd_w) if rfd_w else None,
            },
            "tracking": tracking_stats
            or {
                "enabled": False,
                "tracker": str(args.tracker),
            },
            "state_priority_soft": {
                "enabled": bool(args.state_priority_soft),
                "iou_thresh": float(args.state_priority_iou_thresh),
                "score_gap_max": float(args.state_priority_score_gap_max),
                "relabel_count": int(soft_relabel_count),
            },
            "cvat_video_xml": cvat_video_stats,
        }
        write_json(out_path.with_name(out_path.stem + "_manifest.json"), manifest)
        print(f"Wrote prelabels: {out_path}")
        if mot_out_json is not None:
            print(f"Wrote MOT JSON: {mot_out_json}")
        if cvat_video_xml_out is not None:
            print(f"Wrote CVAT Video XML: {cvat_video_xml_out}")
        print(f"Backend: {backend}; images: {len(coco_images)}; annotations: {len(coco_anns)}")
    finally:
        if temp_images_root is not None:
            shutil.rmtree(temp_images_root, ignore_errors=True)


if __name__ == "__main__":
    main()
