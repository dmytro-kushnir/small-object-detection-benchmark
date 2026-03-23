#!/usr/bin/env python3
"""Compare ANTS stage-1 export (dense ROIs off, no harmful merge) to a reference COCO preds JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_INF_DIR = Path(__file__).resolve().parent
if str(_INF_DIR) not in sys.path:
    sys.path.insert(0, str(_INF_DIR))

from ants_v1.pipeline import run_one_image  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else {}


def _load_gt_name_to_id(coco_gt_path: Path) -> dict[str, int]:
    data = json.loads(coco_gt_path.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for im in data.get("images", []):
        fn = im.get("file_name")
        iid = im.get("id")
        if fn is not None and iid is not None:
            out[Path(str(fn)).name] = int(iid)
    return out


def _norm_list(dets: list[dict[str, Any]]) -> list[tuple[float, float, float, float, float, int]]:
    rows: list[tuple[float, float, float, float, float, int]] = []
    for d in dets:
        bb = d.get("bbox") or []
        if len(bb) < 4:
            continue
        x, y, w, h = map(float, bb[:4])
        rows.append(
            (
                round(x, 3),
                round(y, 3),
                round(x + w, 3),
                round(y + h, 3),
                round(float(d.get("score", 0.0)), 5),
                int(d.get("category_id", 0)),
            )
        )
    rows.sort()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--coco-gt", type=str, required=True)
    p.add_argument("--reference-pred", type=str, required=True, help="e.g. ants_expA002b predictions_val.json")
    p.add_argument("--config", type=str, default="configs/expA004_ants_v1.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument(
        "--tol",
        type=float,
        default=1.0,
        help="Max abs diff per bbox/score field (pixels / score units; default 1.0)",
    )
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics.", file=sys.stderr)
        sys.exit(1)
    import cv2

    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    gt_path = Path(args.coco_gt).expanduser().resolve()
    ref_path = Path(args.reference_pred).expanduser().resolve()
    cfg_path = Path(args.config).expanduser().resolve()

    if not ref_path.is_file():
        print(f"Reference preds not found: {ref_path}", file=sys.stderr)
        sys.exit(1)

    cfg = _load_yaml(cfg_path)
    cfg["enable_dense_rois"] = False
    cfg["pipeline_mode"] = "merged"

    ref_raw = json.loads(ref_path.read_text(encoding="utf-8"))
    ref_by_im: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for d in ref_raw:
        ref_by_im[int(d["image_id"])].append(d)

    name_to_id = _load_gt_name_to_id(gt_path)
    coco = json.loads(gt_path.read_text(encoding="utf-8"))
    work: list[tuple[Path, int, str]] = []
    for im in coco.get("images", []):
        fn = im.get("file_name")
        iid = im.get("id")
        if fn is None or iid is None:
            continue
        base = Path(str(fn)).name
        if base not in name_to_id:
            continue
        ip = source / base
        if ip.is_file():
            work.append((ip, int(name_to_id[base]), base))

    if args.max_images is not None:
        work = work[: max(0, args.max_images)]

    model = YOLO(str(weights))
    mismatches = 0
    compared = 0

    for img_path, im_id, base in work:
        arr = cv2.imread(str(img_path))
        if arr is None:
            continue
        res = run_one_image(model, arr, im_id, base, cfg, device=args.device)
        ants_list = [d for d in res.coco_dets if d["image_id"] == im_id]
        ref_list = ref_by_im.get(im_id, [])

        a = _norm_list(ants_list)
        b = _norm_list(ref_list)
        compared += 1
        if len(a) != len(b):
            print(f"image_id={im_id} {base}: count {len(a)} vs ref {len(b)}")
            mismatches += 1
            continue
        ok = True
        for t1, t2 in zip(a, b):
            for i in range(6):
                if i < 5:
                    if abs(t1[i] - t2[i]) > args.tol:
                        ok = False
                        break
                elif t1[i] != t2[i]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            print(f"image_id={im_id} {base}: first diff\n  ANTS {a[:3]}…\n  REF  {b[:3]}…")
            mismatches += 1

    print(f"Compared {compared} images; mismatches={mismatches}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
