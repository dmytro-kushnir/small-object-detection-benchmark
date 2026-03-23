#!/usr/bin/env python3
"""EXP-A006 visualizations: track overlays + before/after + highlights."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))
import viz_coco_overlays as vco  # noqa: E402


def _draw_track_overlay(
    img: np.ndarray,
    tracks: list[dict[str, Any]],
) -> None:
    for t in tracks:
        x, y, w, h = map(int, t["bbox"])
        tid = int(t["track_id"])
        color = (37 * tid % 255, 97 * tid % 255, 173 * tid % 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            img,
            f"id:{tid}",
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _counts(
    gts: list[dict[str, Any]],
    dets: list[dict[str, Any]],
    score_thr: float,
) -> tuple[int, int]:
    used_g, used_d = vco._match_with_indices(gts, dets, score_thr, 0.5)
    fp = sum(1 for i, d in enumerate(dets) if float(d.get("score", 1.0)) >= score_thr and i not in used_d)
    tp = len(used_d)
    return tp, fp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=str, required=True)
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--pred-before", type=str, required=True, help="EXP-A005 optimized predictions")
    p.add_argument("--pred-after", type=str, required=True, help="EXP-A006 smoothed predictions")
    p.add_argument("--tracks", type=str, required=True, help="tracks JSON from track step")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--score-thr", type=float, default=0.25)
    p.add_argument("--max-images", type=int, default=250)
    args = p.parse_args()

    gt_path = Path(args.gt).expanduser().resolve()
    img_dir = Path(args.images_dir).expanduser().resolve()
    pred_b = Path(args.pred_before).expanduser().resolve()
    pred_a = Path(args.pred_after).expanduser().resolve()
    tracks_p = Path(args.tracks).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    for f in (gt_path, img_dir, pred_b, pred_a, tracks_p):
        if not f.exists():
            print(f"Missing path: {f}", file=sys.stderr)
            sys.exit(1)

    coco = vco._load_coco(gt_path)
    labels = vco._category_names(coco)
    before = vco._load_predictions(pred_b)
    after = vco._load_predictions(pred_a)
    tracks_raw = json.loads(tracks_p.read_text(encoding="utf-8"))
    tracks = tracks_raw.get("tracks") if isinstance(tracks_raw, dict) else tracks_raw
    if not isinstance(tracks, list):
        tracks = []

    # 1) Before/after overlays
    before_dir = out_root / "before_panels"
    after_dir = out_root / "after_panels"
    side_dir = out_root / "before_after"
    tracks_dir = out_root / "tracks_over_time"
    high_dir = out_root / "highlights"

    n_b = vco.run_comparisons(coco, before, img_dir, before_dir, labels, args.score_thr, args.max_images)
    n_a = vco.run_comparisons(coco, after, img_dir, after_dir, labels, args.score_thr, args.max_images)

    # 2) Track overlays
    by_img_tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for t in tracks:
        if t.get("image_id") is None:
            continue
        by_img_tracks[int(t["image_id"])].append(t)
    im_recs = vco._image_records(coco)
    ids = sorted(im_recs.keys())[: args.max_images]
    tracks_dir.mkdir(parents=True, exist_ok=True)
    n_t = 0
    for iid in ids:
        imr = im_recs[iid]
        path = img_dir / Path(str(imr["file_name"])).name
        if not path.is_file():
            continue
        arr = cv2.imread(str(path))
        if arr is None:
            continue
        _draw_track_overlay(arr, by_img_tracks.get(iid, []))
        cv2.imwrite(str(tracks_dir / path.name), arr)
        n_t += 1

    # 3) Side-by-side + highlight stats
    side_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)
    gt_by_im = vco._anns_by_image(coco)
    b_by_im: dict[int, list[dict[str, Any]]] = defaultdict(list)
    a_by_im: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for d in before:
        b_by_im[int(d["image_id"])].append(d)
    for d in after:
        a_by_im[int(d["image_id"])].append(d)
    n_s = 0
    n_h = 0
    for pimg in sorted(before_dir.glob("*.jpg")):
        q = after_dir / pimg.name
        if not q.is_file():
            continue
        a = cv2.imread(str(pimg))
        b = cv2.imread(str(q))
        if a is None or b is None:
            continue
        h = min(a.shape[0], b.shape[0])
        if a.shape[0] != h:
            a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
        if b.shape[0] != h:
            b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
        gap = np.zeros((h, 8, 3), dtype=np.uint8)
        row = np.hstack([a, gap, b])
        cv2.imwrite(str(side_dir / pimg.name), row)
        n_s += 1

        # highlight summary per frame
        iid = None
        for x in ids:
            if Path(str(im_recs[x]["file_name"])).name == pimg.name:
                iid = x
                break
        if iid is None:
            continue
        tp_b, fp_b = _counts(gt_by_im.get(iid, []), b_by_im.get(iid, []), args.score_thr)
        tp_a, fp_a = _counts(gt_by_im.get(iid, []), a_by_im.get(iid, []), args.score_thr)
        canvas = np.zeros((110, 640, 3), dtype=np.uint8)
        txt = [
            f"frame: {pimg.name}",
            f"removed_false_positives: {max(0, fp_b - fp_a)}",
            f"recovered_detections_tp_gain: {max(0, tp_a - tp_b)}",
            f"tracks_on_frame: {len(by_img_tracks.get(iid, []))}",
        ]
        for i, line in enumerate(txt):
            cv2.putText(
                canvas,
                line,
                (8, 24 + 24 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(high_dir / pimg.name), canvas)
        n_h += 1

    print(f"Wrote {n_b} before panels → {before_dir}")
    print(f"Wrote {n_a} after panels → {after_dir}")
    print(f"Wrote {n_t} track overlays → {tracks_dir}")
    print(f"Wrote {n_s} before/after rows → {side_dir}")
    print(f"Wrote {n_h} highlight panels → {high_dir}")


if __name__ == "__main__":
    main()
