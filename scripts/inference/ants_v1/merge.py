"""Merge stage-1 and refined detections with NMS or ROI replacement."""

from __future__ import annotations

from typing import Any, Sequence

from .dense_regions import ROI


# Each det: (x1, y1, x2, y2, score, cls)


def _center_in_rois(cx: float, cy: float, rois: Sequence[ROI]) -> bool:
    """True if (cx, cy) lies in the same half-open rectangle as ``image[y1:y2, x1:x2]``."""
    for r in rois:
        if r.x1 <= cx < r.x2 and r.y1 <= cy < r.y2:
            return True
    return False


def _nms_batched(
    dets: list[tuple[float, float, float, float, float, int]],
    iou_thr: float,
    class_agnostic: bool,
) -> list[tuple[float, float, float, float, float, int]]:
    if not dets:
        return []
    try:
        import torch
        from torchvision.ops import batched_nms
    except ImportError:
        return _nms_python(dets, iou_thr)

    boxes = torch.tensor([[d[0], d[1], d[2], d[3]] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d[4] for d in dets], dtype=torch.float32)
    if class_agnostic:
        idxs = torch.zeros(len(dets), dtype=torch.int64)
    else:
        idxs = torch.tensor([d[5] for d in dets], dtype=torch.int64)
    keep = batched_nms(boxes, scores, idxs, float(iou_thr))
    return [dets[i] for i in keep.tolist()]


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def _nms_python(
    dets: list[tuple[float, float, float, float, float, int]],
    iou_thr: float,
) -> list[tuple[float, float, float, float, float, int]]:
    order = sorted(range(len(dets)), key=lambda i: -dets[i][4])
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [
            j
            for j in order
            if _iou_xyxy(
                (dets[i][0], dets[i][1], dets[i][2], dets[i][3]),
                (dets[j][0], dets[j][1], dets[j][2], dets[j][3]),
            )
            < float(iou_thr)
        ]
    return [dets[i] for i in keep]


def merge_detections(
    stage1: list[tuple[float, float, float, float, float, int]],
    refined: list[tuple[float, float, float, float, float, int]],
    rois: Sequence[ROI],
    cfg: dict[str, Any],
) -> list[tuple[float, float, float, float, float, int]]:
    strategy = str(cfg.get("merge_strategy", "nms")).lower()
    iou_thr = float(cfg.get("nms_iou", 0.45))
    post_nms = bool(cfg.get("enable_post_merge_nms", True))
    class_agnostic = bool(cfg.get("nms_class_agnostic", True))

    def maybe_nms(
        dets: list[tuple[float, float, float, float, float, int]],
    ) -> list[tuple[float, float, float, float, float, int]]:
        if not dets or not post_nms:
            return list(dets)
        return _nms_batched(dets, iou_thr, class_agnostic)

    if strategy == "union":
        return stage1 + refined

    # No refined boxes: never drop stage-1 inside ROIs without replacement; do not
    # re-NMS stage-1 (Ultralytics already NMS'd; matches infer_yolo export).
    if not refined:
        return list(stage1)

    if strategy == "nms_replace_in_roi":
        kept: list[tuple[float, float, float, float, float, int]] = []
        for d in stage1:
            x1, y1, x2, y2, sc, cl = d
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            if _center_in_rois(cx, cy, rois):
                continue
            kept.append(d)
        combined = kept + refined
        return maybe_nms(combined)

    combined = stage1 + refined
    return maybe_nms(combined)


def dets_to_coco_records(
    dets: list[tuple[float, float, float, float, float, int]],
    image_id: int,
    img_w: int,
    img_h: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for x1, y1, x2, y2, score, cls in dets:
        x1 = max(0.0, min(float(x1), float(img_w)))
        y1 = max(0.0, min(float(y1), float(img_h)))
        x2 = max(0.0, min(float(x2), float(img_w)))
        y2 = max(0.0, min(float(y2), float(img_h)))
        if x2 <= x1 or y2 <= y1:
            continue
        w, h = x2 - x1, y2 - y1
        out.append(
            {
                "image_id": int(image_id),
                "category_id": int(cls),
                "bbox": [x1, y1, w, h],
                "score": float(score),
            }
        )
    return out


def coco_list_to_xyxy(
    records: list[dict[str, Any]],
) -> list[tuple[float, float, float, float, float, int]]:
    """COCO bbox list [x,y,w,h] → internal xyxy + score + cls."""
    out: list[tuple[float, float, float, float, float, int]] = []
    for r in records:
        bb = r.get("bbox")
        if not bb or len(bb) < 4:
            continue
        x, y, w, h = map(float, bb[:4])
        sc = float(r.get("score", 1.0))
        cid = int(r.get("category_id", 0))
        out.append((x, y, x + w, y + h, sc, cid))
    return out
