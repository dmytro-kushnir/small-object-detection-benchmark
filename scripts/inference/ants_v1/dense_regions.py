"""Dense ROI detection from stage-1 box centers (grid or DBSCAN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ROI:
    """Axis-aligned ROI in full-image pixel coordinates.

    ``(x1, y1)`` is the first included column/row; ``(x2, y2)`` is **exclusive**
    (same as NumPy ``image[y1:y2, x1:x2]``). Values are integer-clamped to the image.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


def _clamp_roi(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> ROI:
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return ROI(x1, y1, x2, y2)


def _centers_from_boxes_xyxy(boxes: Sequence[tuple[float, float, float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in boxes:
        out.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
    return out


def dense_rois_grid(
    img_w: int,
    img_h: int,
    boxes_xyxy: Sequence[tuple[float, float, float, float]],
    rows: int,
    cols: int,
    count_threshold: int,
    pad_frac: float,
) -> list[ROI]:
    """Assign box centers to a rows×cols grid; union cells with count > threshold."""
    if img_w <= 0 or img_h <= 0 or rows < 1 or cols < 1:
        return []
    if not boxes_xyxy:
        return []

    cell_w = img_w / cols
    cell_h = img_h / rows
    counts: list[list[int]] = [[0 for _ in range(cols)] for _ in range(rows)]

    for x1, y1, x2, y2 in boxes_xyxy:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        j = int(cx / cell_w)
        i = int(cy / cell_h)
        j = max(0, min(j, cols - 1))
        i = max(0, min(i, rows - 1))
        counts[i][j] += 1

    dense_cells: list[tuple[int, int]] = []
    for i in range(rows):
        for j in range(cols):
            if counts[i][j] > count_threshold:
                dense_cells.append((i, j))

    if not dense_cells:
        return []

    dense_set = set(dense_cells)
    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []

    for start in dense_cells:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            i, j = stack.pop()
            comp.append((i, j))
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= ni < rows and 0 <= nj < cols and (ni, nj) in dense_set:
                    if (ni, nj) not in visited:
                        visited.add((ni, nj))
                        stack.append((ni, nj))
        components.append(comp)

    rois: list[ROI] = []
    for comp in components:
        rx1 = min(j * cell_w for i, j in comp)
        ry1 = min(i * cell_h for i, j in comp)
        rx2 = max((j + 1) * cell_w for i, j in comp)
        ry2 = max((i + 1) * cell_h for i, j in comp)
        ix1, iy1, ix2, iy2 = int(rx1), int(ry1), int(rx2), int(ry2)
        rw = ix2 - ix1
        rh = iy2 - iy1
        pad = int(max(rw, rh) * float(pad_frac))
        ix1 -= pad
        iy1 -= pad
        ix2 += pad
        iy2 += pad
        rois.append(_clamp_roi(ix1, iy1, ix2, iy2, img_w, img_h))
    return rois


def dense_rois_dbscan(
    img_w: int,
    img_h: int,
    boxes_xyxy: Sequence[tuple[float, float, float, float]],
    eps_px: float,
    min_samples: int,
    pad_frac: float,
) -> list[ROI]:
    """Cluster box centers with DBSCAN; one padded bbox per cluster."""
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as e:
        raise ImportError(
            "roi_mode dbscan requires scikit-learn (pip install scikit-learn)."
        ) from e

    if not boxes_xyxy:
        return []
    centers = _centers_from_boxes_xyxy(boxes_xyxy)
    if len(centers) < min_samples:
        return []

    import numpy as np

    X = np.asarray(centers, dtype=np.float64)
    labels = DBSCAN(eps=float(eps_px), min_samples=int(min_samples)).fit_predict(X)
    rois: list[ROI] = []
    for lab in sorted(set(labels.tolist())):
        if lab < 0:
            continue
        pts = X[labels == lab]
        cx1, cy1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        cx2, cy2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        ix1, iy1 = int(cx1), int(cy1)
        ix2, iy2 = int(cx2), int(cy2)
        rw = ix2 - ix1
        rh = iy2 - iy1
        pad = int(max(rw, rh, 1) * float(pad_frac))
        ix1 -= pad
        iy1 -= pad
        ix2 += pad
        iy2 += pad
        rois.append(_clamp_roi(ix1, iy1, ix2, iy2, img_w, img_h))
    return rois


def dense_rois_from_config(
    img_w: int,
    img_h: int,
    boxes_xyxy: Sequence[tuple[float, float, float, float]],
    cfg: dict[str, Any],
) -> list[ROI]:
    if not bool(cfg.get("enable_dense_rois", True)):
        return []
    mode = str(cfg.get("roi_mode", "grid")).lower()
    pad = float(cfg.get("roi_pad_frac", 0.15))
    if mode == "dbscan":
        return dense_rois_dbscan(
            img_w,
            img_h,
            boxes_xyxy,
            float(cfg.get("dbscan_eps_px", 48.0)),
            int(cfg.get("dbscan_min_samples", 6)),
            pad,
        )
    return dense_rois_grid(
        img_w,
        img_h,
        boxes_xyxy,
        int(cfg.get("grid_rows", 6)),
        int(cfg.get("grid_cols", 6)),
        int(cfg.get("grid_count_threshold", 4)),
        pad,
    )
