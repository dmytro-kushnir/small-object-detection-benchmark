#!/usr/bin/env python3
"""EXP-A004: ANTS v1 — dense-region refinement inference → COCO JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_INF_DIR = Path(__file__).resolve().parent
if str(_INF_DIR) not in sys.path:
    sys.path.insert(0, str(_INF_DIR))

from ants_v1.pipeline import RunningStats, run_one_image  # noqa: E402

from tqdm import tqdm


def _load_gt_name_to_id(coco_gt_path: Path) -> dict[str, int]:
    data = json.loads(coco_gt_path.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for im in data.get("images", []):
        fn = im.get("file_name")
        iid = im.get("id")
        if fn is not None and iid is not None:
            out[Path(str(fn)).name] = int(iid)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True, help="Val images directory")
    p.add_argument("--coco-gt", type=str, required=True)
    p.add_argument(
        "--out",
        type=str,
        default="experiments/yolo/ants_expA004/predictions_val.json",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/expA004_ants_v1.yaml",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--rois-out",
        type=str,
        default="experiments/yolo/ants_expA004/rois.json",
    )
    p.add_argument(
        "--stage1-out",
        type=str,
        default="experiments/yolo/ants_expA004/predictions_stage1_val.json",
    )
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument(
        "--no-progress",
        action="store_true",
    )
    p.add_argument(
        "--pipeline-mode",
        type=str,
        default=None,
        choices=["merged", "stage1_only", "union_refined"],
        help="Override config pipeline_mode (merged=full ANTS; stage1_only=skip ROI/refine).",
    )
    p.add_argument(
        "--parity-baseline",
        action="store_true",
        help="Shortcut: enable_dense_rois=false for parity vs vanilla preds JSON.",
    )
    p.add_argument(
        "--dump-refine-viz",
        type=str,
        default=None,
        help="Directory for ROI crop debug images with refined boxes (crop coordinates).",
    )
    p.add_argument(
        "--max-refine-viz-rois",
        type=int,
        default=80,
        help="Max ROI crops to write across the whole run (default 80).",
    )
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics (see requirements.txt).", file=sys.stderr)
        sys.exit(1)

    import cv2

    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    gt_path = Path(args.coco_gt).expanduser().resolve()
    cfg_path = Path(args.config).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    rois_path = Path(args.rois_out).expanduser().resolve()
    st1_path = Path(args.stage1_out).expanduser().resolve()

    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)
    if not source.is_dir():
        print(f"Source dir not found: {source}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.is_file():
        print(f"COCO GT not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = _load_yaml(cfg_path)
    if args.parity_baseline:
        cfg["enable_dense_rois"] = False
    if args.pipeline_mode is not None:
        cfg["pipeline_mode"] = args.pipeline_mode
    name_to_id = _load_gt_name_to_id(gt_path)
    coco = json.loads(gt_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])

    work: list[tuple[Path, int, str]] = []
    for im in images:
        fn = im.get("file_name")
        iid = im.get("id")
        if fn is None or iid is None:
            continue
        base = Path(str(fn)).name
        if base not in name_to_id:
            continue
        img_path = source / base
        if not img_path.is_file():
            continue
        work.append((img_path, int(name_to_id[base]), base))

    if args.max_images is not None:
        work = work[: max(0, args.max_images)]

    model = YOLO(str(weights))

    viz_root = (
        Path(args.dump_refine_viz).expanduser().resolve() if args.dump_refine_viz else None
    )
    viz_budget = max(0, int(args.max_refine_viz_rois))

    all_dets: list[dict[str, Any]] = []
    all_stage1: list[dict[str, Any]] = []
    roi_records: list[dict[str, Any]] = []
    stats = RunningStats()
    missing = 0

    it = work
    if not args.no_progress and work:
        it = tqdm(work, desc="ANTS v1 infer", unit="img", file=sys.stderr)

    for item in it:
        img_path, im_id, base = item
        arr = cv2.imread(str(img_path))
        if arr is None:
            missing += 1
            continue
        res = run_one_image(model, arr, im_id, base, cfg, device=args.device)
        stats.update(res.rois, res.width, res.height)
        if viz_root is not None and viz_budget > 0 and res.refine_debug:
            viz_root.mkdir(parents=True, exist_ok=True)
            for roi, lboxes in res.refine_debug:
                if viz_budget <= 0:
                    break
                crop = arr[roi.y1 : roi.y2, roi.x1 : roi.x2]
                if crop.size == 0:
                    continue
                vis = crop.copy()
                for x1, y1, x2, y2, _, _ in lboxes:
                    cv2.rectangle(
                        vis,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 220, 0),
                        1,
                    )
                fn = f"i{im_id}_r{roi.x1}_{roi.y1}_n{len(lboxes)}.jpg"
                cv2.imwrite(str(viz_root / fn), vis)
                viz_budget -= 1
        all_dets.extend(res.coco_dets)
        all_stage1.extend(res.stage1_coco)
        roi_records.append(
            {
                "image_id": res.image_id,
                "file_name": res.file_name,
                "width": res.width,
                "height": res.height,
                "n_rois": len(res.rois),
                "rois": [r.to_list() for r in res.rois],
            }
        )

    n_img = stats.total_images
    mean_rois = stats.total_rois / n_img if n_img else 0.0
    mean_roi_frac = stats.roi_area_sum / n_img if n_img else 0.0

    rois_payload = {
        "config_path": str(cfg_path),
        "weights": str(weights),
        "images": roi_records,
        "stats": {
            "n_images_processed": n_img,
            "mean_rois_per_image": mean_rois,
            "mean_roi_area_frac_of_image": mean_roi_frac,
            "skipped_unreadable": missing,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_dets, indent=2), encoding="utf-8")
    rois_path.parent.mkdir(parents=True, exist_ok=True)
    rois_path.write_text(json.dumps(rois_payload, indent=2), encoding="utf-8")
    st1_path.parent.mkdir(parents=True, exist_ok=True)
    st1_path.write_text(json.dumps(all_stage1, indent=2), encoding="utf-8")

    meta = {
        "ants_v1_config": str(cfg_path),
        "weights": str(weights),
        "predictions_out": str(out_path),
        "rois_out": str(rois_path),
        "stage1_out": str(st1_path),
    }
    (out_path.parent / "ants_v1_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(
        f"Wrote {out_path} ({len(all_dets)} dets), {rois_path}, {st1_path} ({len(all_stage1)} stage1)",
        flush=True,
    )


if __name__ == "__main__":
    main()
