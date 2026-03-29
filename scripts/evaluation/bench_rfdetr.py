#!/usr/bin/env python3
"""Time RF-DETR predict on val images; write JSON for evaluate.py --inference-benchmark-json."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import json
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from repo_paths import path_for_artifact


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {k: v for k, v in kwargs.items() if v is not None}
    return {k: v for k, v in kwargs.items() if k in params and v is not None}

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_INF = Path(__file__).resolve().parent.parent / "inference"
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

from coco_pred_common import load_gt_filename_to_image_id  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True, help="Val images directory")
    p.add_argument("--coco-gt", type=str, required=True)
    p.add_argument("--model-class", type=str, default="RFDETRSmall")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--out",
        type=str,
        default="experiments/rfdetr/ants_expA005/inference_benchmark.json",
    )
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--config", type=str, default=None, help="Optional YAML path for metadata only")
    args = p.parse_args()

    try:
        rfdetr = importlib.import_module("rfdetr")
    except ImportError:
        print("Install rfdetr.", file=sys.stderr)
        sys.exit(1)

    if not hasattr(rfdetr, args.model_class):
        print(f"Unknown model class {args.model_class}", file=sys.stderr)
        sys.exit(1)
    ModelCls = getattr(rfdetr, args.model_class)
    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    gt_path = Path(args.coco_gt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    model = ModelCls(**_filter_kwargs(ModelCls.__init__, {"pretrain_weights": str(weights)}))
    if args.device:
        dev = args.device.strip()
        if dev.isdigit():
            dev = f"cuda:{dev}"
        to_dev = getattr(model, "to", None)
        if callable(to_dev):
            try:
                import torch

                to_dev(torch.device(dev))
            except Exception:
                pass

    # Optional inference optimization (used for the dedicated "optimized inference" run).
    # This is environment-controlled to keep the unoptimized baseline reproducible.
    opt_flag = os.environ.get("EXP_A005_OPTIMIZE_INFERENCE", "0").strip().lower()
    if opt_flag in {"1", "true", "yes", "on"}:
        opt_fn = getattr(model, "optimize_for_inference", None)
        if callable(opt_fn):
            opt_fn()

    name_to_id = load_gt_filename_to_image_id(gt_path)
    coco = json.loads(gt_path.read_text(encoding="utf-8"))
    work: list[Path] = []
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
            "backend": "rfdetr",
            "note": "No images to bench",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        return

    n_warm = min(args.warmup, len(work))
    kw = _filter_kwargs(
        model.predict,
        {"threshold": float(args.conf), "confidence": float(args.conf)},
    )

    def _run_one(ip: Path) -> None:
        bgr = cv2.imread(str(ip))
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        model.predict(rgb, **kw)

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
            "backend": "rfdetr",
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
        "backend": "rfdetr",
        "model_class": args.model_class,
        "conf": float(args.conf),
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
