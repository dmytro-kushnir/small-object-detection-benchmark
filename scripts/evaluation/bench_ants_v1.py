#!/usr/bin/env python3
"""Time full ANTS v1 pipeline per val image; write JSON for evaluate.py --inference-benchmark-json."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_INF_DIR = Path(__file__).resolve().parent.parent / "inference"
if str(_INF_DIR) not in sys.path:
    sys.path.insert(0, str(_INF_DIR))

from ants_v1.pipeline import run_one_image  # noqa: E402


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
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--coco-gt", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/expA004_ants_v1.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--out",
        type=str,
        default="experiments/yolo/ants_expA004/inference_benchmark.json",
    )
    p.add_argument("--max-images", type=int, default=None)
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
    cfg_path = Path(args.config).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    cfg = _load_yaml(cfg_path)
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

    if not work:
        payload = {
            "fps": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "n_images": 0,
            "warmup": args.warmup,
            "backend": "ants_v1",
            "note": "No images to bench",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        return

    model = YOLO(str(weights))
    n_warm = min(args.warmup, len(work))
    for img_path, im_id, base in work[:n_warm]:
        arr = cv2.imread(str(img_path))
        if arr is None:
            continue
        run_one_image(model, arr, im_id, base, cfg, device=args.device)

    to_time = work[n_warm:]
    if not to_time:
        payload = {
            "fps": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "n_images": 0,
            "warmup": args.warmup,
            "backend": "ants_v1",
            "note": "All images used for warmup",
            "config_path": str(cfg_path),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        return

    times: list[float] = []
    for img_path, im_id, base in to_time:
        arr = cv2.imread(str(img_path))
        if arr is None:
            continue
        t0 = time.perf_counter()
        run_one_image(model, arr, im_id, base, cfg, device=args.device)
        times.append(time.perf_counter() - t0)

    if not times:
        payload = {
            "fps": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "n_images": 0,
            "warmup": args.warmup,
            "backend": "ants_v1",
            "note": "No successful timed images",
            "config_path": str(cfg_path),
        }
    else:
        mean_s = sum(times) / len(times)
        var = sum((t - mean_s) ** 2 for t in times) / len(times)
        std_s = math.sqrt(var)
        payload = {
            "fps": len(times) / sum(times) if sum(times) > 0 else None,
            "latency_ms_mean": mean_s * 1000.0,
            "latency_ms_std": std_s * 1000.0,
            "n_images": len(times),
            "warmup": args.warmup,
            "backend": "ants_v1",
            "config_path": str(cfg_path),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
