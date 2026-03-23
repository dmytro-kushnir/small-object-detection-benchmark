#!/usr/bin/env python3
"""SAHI sliced inference with Ultralytics YOLO weights → COCO list JSON (evaluate.py compatible)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

_INF = Path(__file__).resolve().parent
if str(_INF) not in sys.path:
    sys.path.insert(0, str(_INF))

from coco_pred_common import load_gt_filename_to_image_id, write_coco_predictions_json


def _load_sahi_bench_module() -> Any:
    p = Path(__file__).resolve().parent.parent / "evaluation" / "sahi_bench.py"
    spec = importlib.util.spec_from_file_location("sahi_bench", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else {}


def _merge_sahi_config(
    yaml_path: Path,
    overrides: dict[str, Any | None],
) -> dict[str, Any]:
    cfg = _load_yaml(yaml_path)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(
        description="SAHI sliced YOLO inference → COCO results JSON for pycocotools."
    )
    p.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    p.add_argument("--source", type=str, required=True, help="Directory of val images")
    p.add_argument(
        "--coco-gt",
        type=str,
        required=True,
        help="COCO GT JSON to map file_name → image_id",
    )
    p.add_argument("--out", type=str, required=True, help="Output predictions JSON path")
    p.add_argument(
        "--sahi-config",
        type=str,
        default="configs/exp003_sahi.yaml",
        help="YAML: slice_*, overlap_*, confidence_threshold, model_type (ultralytics), yolo_imgsz, optional SAHI merge keys",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='Ultralytics/SAHI device: "0", "cuda:0", "cpu", or omit for ANTS_DEVICE / SMOKE_DEVICE / auto CUDA.',
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bar (e.g. logs only); use with --progress-every for sparse prints.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
        metavar="N",
        help="With --no-progress, print every N images (0 = silent). Ignored when tqdm is enabled.",
    )
    p.add_argument("--slice-height", type=int, default=None)
    p.add_argument("--slice-width", type=int, default=None)
    p.add_argument("--overlap-height-ratio", type=float, default=None)
    p.add_argument("--overlap-width-ratio", type=float, default=None)
    p.add_argument("--confidence-threshold", type=float, default=None)
    p.add_argument("--yolo-imgsz", type=int, default=None)
    args = p.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    source_path = Path(args.source).expanduser().resolve()
    gt_path = Path(args.coco_gt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    cfg_path = Path(args.sahi_config).expanduser().resolve()

    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)
    if not source_path.is_dir():
        print(f"Source dir not found: {source_path}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.is_file():
        print(f"COCO GT not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    if not cfg_path.is_file():
        print(f"SAHI config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    sahi_params = _merge_sahi_config(
        cfg_path,
        {
            "slice_height": args.slice_height,
            "slice_width": args.slice_width,
            "overlap_height_ratio": args.overlap_height_ratio,
            "overlap_width_ratio": args.overlap_width_ratio,
            "confidence_threshold": args.confidence_threshold,
            "yolo_imgsz": args.yolo_imgsz,
        },
    )

    try:
        sb = _load_sahi_bench_module()
    except Exception as e:
        print(f"Failed to load sahi_bench: {e}", file=sys.stderr)
        sys.exit(1)

    resolved_dev = sb.resolve_sahi_device(args.device)
    try:
        cuda_ok = False
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
        except ImportError:
            pass
        print(
            f"infer_sahi_yolo: device={resolved_dev!r} (torch.cuda_available={cuda_ok})",
            flush=True,
        )
        model = sb.build_sahi_detection_model(weights, args.device, sahi_params)
    except ImportError:
        print("Install sahi (pip install -r requirements.txt).", file=sys.stderr)
        sys.exit(1)

    name_to_id = load_gt_filename_to_image_id(gt_path)
    coco = json.loads(gt_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])

    work: list[tuple[Path, int]] = []
    missing: set[str] = set()
    for im in images:
        fn = im.get("file_name")
        iid = im.get("id")
        if fn is None or iid is None:
            continue
        base = Path(str(fn)).name
        if base not in name_to_id:
            missing.add(base)
            continue
        img_path = source_path / base
        if not img_path.is_file():
            missing.add(base)
            continue
        im_id = int(name_to_id[base])
        work.append((img_path, im_id))

    detections: list[dict[str, Any]] = []
    n_work = len(work)
    use_tqdm = not args.no_progress and n_work > 0
    it = work
    if use_tqdm:
        it = tqdm(work, desc="SAHI infer", unit="img", file=sys.stderr)
    for i, (img_path, im_id) in enumerate(it, start=1):
        result = sb.run_sahi_sliced_on_path(img_path, model, sahi_params)
        detections.extend(sb.predictions_to_coco_dets(result, im_id))
        if (
            not use_tqdm
            and args.progress_every > 0
            and i % args.progress_every == 0
        ):
            print(f"infer_sahi_yolo: {i}/{n_work} images", flush=True)

    if missing:
        print(
            f"Warning: skipped {len(missing)} missing/unknown files "
            f"(showing up to 5): {sorted(missing)[:5]}",
            file=sys.stderr,
        )

    write_coco_predictions_json(out_path, detections)

    meta = {
        "sahi_params": sahi_params,
        "sahi_config_path": str(cfg_path),
        "weights": str(weights),
        "coco_gt": str(gt_path),
        "source": str(source_path),
        "predictions_out": str(out_path),
        "device_resolved": resolved_dev,
    }
    (out_path.parent / "sahi_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path} ({len(detections)} detections)")
    print(f"Wrote {out_path.parent / 'sahi_config.json'}")


if __name__ == "__main__":
    main()
