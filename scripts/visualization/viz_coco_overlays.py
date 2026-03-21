#!/usr/bin/env python3
"""COCO prediction/GT overlays from existing predictions JSON (no train/eval)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def _load_coco(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "annotations" in raw:
        return list(raw["annotations"])
    raise ValueError("Predictions must be a JSON list or dict with 'annotations'")


def _category_names(coco: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for c in coco.get("categories", []):
        out[int(c["id"])] = str(c.get("name", c["id"]))
    return out


def _anns_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    m: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in coco.get("annotations", []):
        m[int(a["image_id"])].append(a)
    return m


def _image_records(coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(im["id"]): im for im in coco.get("images", [])}


def _resolve_image_path(images_dir: Path, file_name: str) -> Path:
    return images_dir / Path(file_name).name


def _draw_boxes(
    img: np.ndarray,
    boxes_xywh: list[tuple[list[float], int, tuple[int, int, int], int]],
    labels: dict[int, str] | None,
) -> None:
    """boxes_xywh: (bbox, cat_id, bgr_color, thickness)."""
    for bbox, cat_id, color, th in boxes_xywh:
        x, y, w, h = map(int, bbox[:4])
        cv2.rectangle(img, (x, y), (x + w, y + h), color, th)
        if labels:
            name = labels.get(cat_id, str(cat_id))
            cv2.putText(
                img,
                name[:20],
                (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )


def _match_with_indices(
    gts: list[dict[str, Any]],
    dets: list[dict[str, Any]],
    score_thr: float,
    iou_thr: float,
) -> tuple[set[int], set[int]]:
    """Return set of GT indices and set of det indices (into dets) that matched."""
    order = sorted(
        range(len(dets)),
        key=lambda i: -float(dets[i].get("score", 1.0)),
    )
    used_g: set[int] = set()
    used_d: set[int] = set()
    for i in order:
        if float(dets[i].get("score", 1.0)) < score_thr:
            continue
        det = dets[i]
        best_j = -1
        best_iou = 0.0
        db = det["bbox"]
        dc = int(det["category_id"])
        for gj, ga in enumerate(gts):
            if gj in used_g:
                continue
            if int(ga["category_id"]) != dc:
                continue
            iou = _iou_xywh(db, ga["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_j = gj
        if best_j >= 0 and best_iou >= iou_thr:
            used_g.add(best_j)
            used_d.add(i)
    return used_g, used_d


def run_predictions_overlay(
    coco: dict[str, Any],
    preds: list[dict[str, Any]],
    images_dir: Path,
    out_dir: Path,
    labels: dict[int, str],
    max_images: int | None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    im_recs = _image_records(coco)
    gt_by_im = _anns_by_image(coco)
    pr_by_im: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        pr_by_im[int(p["image_id"])].append(p)

    ids = sorted(im_recs.keys())
    if max_images is not None:
        ids = ids[:max_images]
    n = 0
    green = (0, 200, 0)
    red = (0, 0, 255)
    for iid in ids:
        imr = im_recs[iid]
        path = _resolve_image_path(images_dir, imr["file_name"])
        if not path.is_file():
            continue
        arr = cv2.imread(str(path))
        if arr is None:
            continue
        boxes: list[tuple[list[float], int, tuple[int, int, int], int]] = []
        for ga in gt_by_im.get(iid, []):
            boxes.append((ga["bbox"], int(ga["category_id"]), green, 2))
        for pa in pr_by_im.get(iid, []):
            boxes.append((pa["bbox"], int(pa["category_id"]), red, 2))
        _draw_boxes(arr, boxes, labels)
        out_path = out_dir / path.name
        cv2.imwrite(str(out_path), arr)
        n += 1
    return n


def run_comparisons(
    coco: dict[str, Any],
    preds: list[dict[str, Any]],
    images_dir: Path,
    out_dir: Path,
    labels: dict[int, str],
    score_thr: float,
    max_images: int | None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    im_recs = _image_records(coco)
    gt_by_im = _anns_by_image(coco)
    pr_by_im: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        pr_by_im[int(p["image_id"])].append(p)

    orange = (0, 140, 255)  # FN — missed GT
    fp_red = (0, 0, 255)
    tp_gt = (80, 180, 80)
    tp_pr = (200, 200, 80)

    ids = sorted(im_recs.keys())
    if max_images is not None:
        ids = ids[:max_images]
    n = 0
    for iid in ids:
        imr = im_recs[iid]
        path = _resolve_image_path(images_dir, imr["file_name"])
        if not path.is_file():
            continue
        arr = cv2.imread(str(path))
        if arr is None:
            continue
        gts = gt_by_im.get(iid, [])
        dets = pr_by_im.get(iid, [])
        used_g, used_d = _match_with_indices(gts, dets, score_thr, 0.5)
        boxes: list[tuple[list[float], int, tuple[int, int, int], int]] = []
        for gi, ga in enumerate(gts):
            if gi in used_g:
                boxes.append((ga["bbox"], int(ga["category_id"]), tp_gt, 1))
            else:
                boxes.append((ga["bbox"], int(ga["category_id"]), orange, 3))
        for di, pa in enumerate(dets):
            if float(pa.get("score", 1.0)) < score_thr:
                continue
            if di in used_d:
                boxes.append((pa["bbox"], int(pa["category_id"]), tp_pr, 1))
            else:
                boxes.append((pa["bbox"], int(pa["category_id"]), fp_red, 3))
        _draw_boxes(arr, boxes, labels)
        # Legend strip (small text)
        cv2.putText(
            arr,
            "Orange thick=FN GT  Red thick=FP  Dimmer=TP",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / path.name), arr)
        n += 1
    return n


def run_dataset_samples(
    coco: dict[str, Any],
    images_dir: Path,
    out_dir: Path,
    labels: dict[int, str],
    num_samples: int,
    seed: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    im_recs = list(_image_records(coco).values())
    if not im_recs:
        return 0
    rng = random.Random(seed)
    rng.shuffle(im_recs)
    im_recs = im_recs[:num_samples]
    gt_by_im = _anns_by_image(coco)
    green = (0, 200, 0)
    n = 0
    for imr in im_recs:
        iid = int(imr["id"])
        path = _resolve_image_path(images_dir, imr["file_name"])
        if not path.is_file():
            continue
        arr = cv2.imread(str(path))
        if arr is None:
            continue
        boxes: list[tuple[list[float], int, tuple[int, int, int], int]] = []
        for ga in gt_by_im.get(iid, []):
            boxes.append((ga["bbox"], int(ga["category_id"]), green, 2))
        _draw_boxes(arr, boxes, labels)
        cv2.imwrite(str(out_dir / path.name), arr)
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="COCO overlays from existing preds + GT.")
    p.add_argument("--pred", type=str, required=True, help="Predictions JSON (list)")
    p.add_argument("--gt", type=str, required=True, help="COCO GT JSON (e.g. instances_val.json)")
    p.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing image files referenced by GT file_name",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="experiments/visualizations/test_run",
        help="Root output directory",
    )
    p.add_argument(
        "--dataset-gt",
        type=str,
        default=None,
        help="Optional COCO JSON for dataset/ samples (default: --gt)",
    )
    p.add_argument(
        "--dataset-images-dir",
        type=str,
        default=None,
        help="Optional images dir for dataset/ samples (default: --images-dir)",
    )
    p.add_argument("--max-images", type=int, default=None, help="Cap images for pred+comparison")
    p.add_argument("--num-dataset-samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-thr", type=float, default=0.25)
    args = p.parse_args()

    pred_path = Path(args.pred).expanduser().resolve()
    gt_path = Path(args.gt).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()

    if not pred_path.is_file():
        print(f"Predictions not found: {pred_path}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.is_file():
        print(f"GT not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"Images dir not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    coco = _load_coco(gt_path)
    preds = _load_predictions(pred_path)
    labels = _category_names(coco)

    ds_gt = Path(args.dataset_gt).expanduser().resolve() if args.dataset_gt else gt_path
    ds_im = Path(args.dataset_images_dir).expanduser().resolve() if args.dataset_images_dir else images_dir
    coco_ds = _load_coco(ds_gt) if ds_gt != gt_path else coco

    sub_pred = out_root / "predictions"
    sub_cmp = out_root / "comparisons"
    sub_ds = out_root / "dataset"

    n1 = run_predictions_overlay(coco, preds, images_dir, sub_pred, labels, args.max_images)
    n2 = run_comparisons(
        coco, preds, images_dir, sub_cmp, labels, args.score_thr, args.max_images
    )
    if not ds_im.is_dir():
        print(f"Dataset images dir not found: {ds_im}", file=sys.stderr)
        n3 = 0
    else:
        n3 = run_dataset_samples(
            coco_ds, ds_im, sub_ds, labels, args.num_dataset_samples, args.seed
        )

    print(
        f"Wrote {n1} prediction overlays → {sub_pred}\n"
        f"       {n2} comparison images → {sub_cmp}\n"
        f"       {n3} dataset samples → {sub_ds}"
    )


if __name__ == "__main__":
    main()
