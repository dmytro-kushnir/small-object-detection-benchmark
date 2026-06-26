#!/usr/bin/env python3
"""Time Faster R-CNN forward on val/test images; JSON for evaluate.py --inference-benchmark-json."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

_SCRIPTS = Path(__file__).resolve().parents[1]
_INF = _SCRIPTS / "inference"
for _p in (_SCRIPTS, _INF):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coco_pred_common import load_gt_filename_to_image_id  # noqa: E402
from faster_rcnn_common import load_faster_rcnn_checkpoint, resolve_device  # noqa: E402
from repo_paths import path_for_artifact  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True, help="Images directory")
    p.add_argument("--coco-gt", type=str, required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--min-size", type=int, default=896)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out", type=str, default="experiments/faster_rcnn/inference_benchmark.json")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--config", type=str, default=None, help="Optional YAML for metadata")
    args = p.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    gt_path = Path(args.coco_gt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    device = resolve_device(args.device)

    model = load_faster_rcnn_checkpoint(
        weights,
        min_size=args.min_size,
        max_size=args.max_size,
        device=device,
    )

    name_to_id = load_gt_filename_to_image_id(gt_path)
    coco = json.loads(gt_path.read_text(encoding="utf-8"))
    work: list[Path] = []
    for im in coco.get("images", []):
        fn = im.get("file_name")
        if fn is None:
            continue
        base = Path(str(fn)).name
        if base not in name_to_id:
            continue
        ip = source / base
        if ip.is_file():
            work.append(ip)

    if args.max_images is not None:
        work = work[: max(0, args.max_images)]

    if not work:
        payload = {
            "fps": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "n_images": 0,
            "warmup": args.warmup,
            "backend": "faster_rcnn",
            "note": "No images to bench",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        return

    conf = float(args.conf)

    def _run_one(ip: Path) -> None:
        bgr = cv2.imread(str(ip))
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(Image.fromarray(rgb)).to(device)
        with torch.no_grad():
            out = model([tensor])[0]
        _ = (out["scores"] >= conf).sum()

    n_warm = min(args.warmup, len(work))
    for ip in work[:n_warm]:
        _run_one(ip)

    to_time = work[n_warm:]
    if not to_time:
        payload = {
            "fps": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "n_images": 0,
            "warmup": args.warmup,
            "backend": "faster_rcnn",
            "note": "All images used for warmup",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        return

    times: list[float] = []
    for ip in to_time:
        t0 = time.perf_counter()
        _run_one(ip)
        times.append(time.perf_counter() - t0)

    mean_s = float(np.mean(times)) if times else 0.0
    std_s = float(np.std(times)) if times else 0.0
    n = len(times)
    fps = n / sum(times) if sum(times) > 0 else None

    payload: dict[str, Any] = {
        "fps": fps,
        "latency_ms_mean": mean_s * 1000.0,
        "latency_ms_std": std_s * 1000.0,
        "n_images": n,
        "warmup": args.warmup,
        "backend": "faster_rcnn",
        "conf": conf,
        "min_size": args.min_size,
        "max_size": args.max_size,
    }
    if args.config:
        cfg_p = Path(args.config).expanduser().resolve()
        if cfg_p.is_file():
            payload["config_path"] = path_for_artifact(cfg_p, _ROOT)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
