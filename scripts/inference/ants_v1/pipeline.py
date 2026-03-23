"""Per-image ANTS v1: stage-1 → ROIs → refine crops → merge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dense_regions import ROI, dense_rois_from_config
from .merge import dets_to_coco_records, merge_detections


@dataclass
class ImagePipelineResult:
    image_id: int
    file_name: str
    width: int
    height: int
    rois: list[ROI]
    stage1_xyxy: list[tuple[float, float, float, float, float, int]]
    refined_xyxy: list[tuple[float, float, float, float, float, int]]
    merged_xyxy: list[tuple[float, float, float, float, float, int]]
    coco_dets: list[dict[str, Any]]
    stage1_coco: list[dict[str, Any]]
    refine_debug: list[tuple[ROI, list[tuple[float, float, float, float, float, int]]]] = field(
        default_factory=list
    )


@dataclass
class RunningStats:
    total_rois: int = 0
    total_images: int = 0
    roi_area_sum: float = 0.0

    def update(self, rois: list[ROI], img_w: int, img_h: int) -> None:
        self.total_images += 1
        self.total_rois += len(rois)
        im_area = max(1, img_w * img_h)
        for r in rois:
            a = max(0, r.x2 - r.x1) * max(0, r.y2 - r.y1)
            self.roi_area_sum += a / im_area


def _boxes_from_result(
    result: Any,
    min_score: float,
) -> list[tuple[float, float, float, float, float, int]]:
    """Boxes after Ultralytics predict; optional min_score (e.g. refine_min_score)."""
    out: list[tuple[float, float, float, float, float, int]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    for b in result.boxes:
        sc = float(b.conf[0]) if b.conf is not None else 1.0
        if sc < min_score:
            continue
        xyxy = b.xyxy[0].tolist()
        x1, y1, x2, y2 = map(float, xyxy)
        cls = int(b.cls[0]) if b.cls is not None else 0
        out.append((x1, y1, x2, y2, sc, cls))
    return out


def _predict_kw(
    cfg: dict[str, Any],
    device: str | None,
    imgsz: int,
    conf: float,
) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "save": False,
        "verbose": False,
        "imgsz": int(imgsz),
        "conf": float(conf),
    }
    if cfg.get("predict_iou") is not None:
        kw["iou"] = float(cfg["predict_iou"])
    if device is not None:
        kw["device"] = device
    return kw


def run_one_image(
    model: Any,
    image_bgr: np.ndarray,
    image_id: int,
    file_name: str,
    cfg: dict[str, Any],
    device: str | None = None,
) -> ImagePipelineResult:
    h, w = image_bgr.shape[:2]
    base_imgsz = int(cfg.get("base_imgsz", 768))
    refine_imgsz = int(cfg.get("refine_imgsz", 896))
    c1 = float(cfg.get("conf_threshold_stage1", 0.25))
    c2 = float(cfg.get("conf_threshold_refine", 0.25))
    refine_min = float(cfg.get("refine_min_score", 0.0))
    pipeline_mode = str(cfg.get("pipeline_mode", "merged")).lower()

    pred_kw = _predict_kw(cfg, device, base_imgsz, c1)
    r1 = model.predict(source=image_bgr, **pred_kw)
    if not r1:
        stage1: list[tuple[float, float, float, float, float, int]] = []
    else:
        stage1 = _boxes_from_result(r1[0], 0.0)

    refine_debug: list[tuple[ROI, list[tuple[float, float, float, float, float, int]]]] = []
    refined: list[tuple[float, float, float, float, float, int]] = []
    rois: list[ROI] = []

    if pipeline_mode == "stage1_only":
        merged = list(stage1)
    else:
        rois = dense_rois_from_config(w, h, [s[:4] for s in stage1], cfg)
        pred_kw2 = _predict_kw(cfg, device, refine_imgsz, c2)
        for roi in rois:
            # Half-open ROI matches merge._center_in_rois (exclusive x2, y2).
            crop = image_bgr[roi.y1 : roi.y2, roi.x1 : roi.x2]
            if crop.size == 0:
                refine_debug.append((roi, []))
                continue
            r2 = model.predict(source=crop, **pred_kw2)
            if not r2:
                refine_debug.append((roi, []))
                continue
            local = _boxes_from_result(r2[0], refine_min)
            refine_debug.append((roi, list(local)))
            for x1, y1, x2, y2, sc, cl in local:
                xf1 = x1 + roi.x1
                yf1 = y1 + roi.y1
                xf2 = x2 + roi.x1
                yf2 = y2 + roi.y1
                xf1 = max(0.0, min(xf1, float(w)))
                xf2 = max(0.0, min(xf2, float(w)))
                yf1 = max(0.0, min(yf1, float(h)))
                yf2 = max(0.0, min(yf2, float(h)))
                refined.append((xf1, yf1, xf2, yf2, sc, cl))

        if pipeline_mode == "union_refined":
            ucfg = {**cfg, "merge_strategy": "union"}
            merged = merge_detections(stage1, refined, rois, ucfg)
        else:
            merged = merge_detections(stage1, refined, rois, cfg)

    coco = dets_to_coco_records(merged, image_id, w, h)
    st_coco = dets_to_coco_records(stage1, image_id, w, h)

    return ImagePipelineResult(
        image_id=image_id,
        file_name=file_name,
        width=w,
        height=h,
        rois=rois,
        stage1_xyxy=stage1,
        refined_xyxy=refined,
        merged_xyxy=merged,
        coco_dets=coco,
        stage1_coco=st_coco,
        refine_debug=refine_debug,
    )
