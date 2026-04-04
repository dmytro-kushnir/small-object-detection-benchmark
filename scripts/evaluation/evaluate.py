#!/usr/bin/env python3
"""COCO detection evaluation: pycocotools COCOeval + P/R @ IoU=0.5 + FPS/latency (Ultralytics).

Extra GT annotation fields (e.g. ``track_id`` on Camponotus exports) are ignored; only bbox
detection metrics are computed.

Output JSON includes ``coco_eval_per_category`` (COCOeval with ``params.catIds`` per class) and
``matched_pr_per_category`` (same greedy IoU≥0.5 / score≥0.25 rule as ``matched_pr``, per class;
images are the union of GT and predictions for that class).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from repo_paths import path_for_artifact


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "annotations" in raw:
        return list(raw["annotations"])
    raise ValueError(f"Predictions must be a JSON list or {{'annotations': [...]}}, got {type(raw)}")


def _greedy_tp_fp_fn_iou50(
    img_to_gts: dict[int, list[dict[str, Any]]],
    img_to_dets: dict[int, list[dict[str, Any]]],
    image_ids: list[int],
) -> tuple[int, int, int]:
    """Greedy match per image @ IoU>=0.5, same category (gts/dets already filtered)."""
    tp = fp = 0
    fn = 0
    for img_id in image_ids:
        gts = img_to_gts.get(img_id, [])
        dets = img_to_dets.get(img_id, [])
        used_gt = [False] * len(gts)
        for det in dets:
            best_i = -1
            best_iou = 0.0
            db = det["bbox"]
            dc = int(det["category_id"])
            for gi, ga in enumerate(gts):
                if used_gt[gi]:
                    continue
                if int(ga["category_id"]) != dc:
                    continue
                iou = _iou_xywh(db, ga["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_i = gi
            if best_i >= 0 and best_iou >= 0.5:
                used_gt[best_i] = True
                tp += 1
            else:
                fp += 1
        fn += sum(1 for u in used_gt if not u)
    return tp, fp, fn


def _precision_recall_iou50(
    gt: Any,
    detections: list[dict[str, Any]],
    score_thr: float = 0.25,
    category_id: int | None = None,
) -> tuple[float, float, int, int, int]:
    """Greedy match per image @ IoU>=0.5, same category; return P, R, TP, FP, FN.

    If ``category_id`` is set, only that class contributes; image set is the union of
    images with GT or predictions for that class (so pred-only FPs count).
    """
    img_to_gts: dict[int, list[dict[str, Any]]] = {}
    for ann in gt.dataset["annotations"]:
        if category_id is not None and int(ann["category_id"]) != category_id:
            continue
        img_to_gts.setdefault(int(ann["image_id"]), []).append(ann)
    img_to_dets: dict[int, list[dict[str, Any]]] = {}
    for d in detections:
        if float(d.get("score", 1.0)) < score_thr:
            continue
        if category_id is not None and int(d["category_id"]) != category_id:
            continue
        img_to_dets.setdefault(int(d["image_id"]), []).append(d)
    for lst in img_to_dets.values():
        lst.sort(key=lambda x: -float(x.get("score", 1.0)))

    if category_id is None:
        image_ids = list(img_to_gts.keys())
    else:
        image_ids = sorted(set(img_to_gts) | set(img_to_dets))

    tp, fp, fn = _greedy_tp_fp_fn_iou50(img_to_gts, img_to_dets, image_ids)
    denom_p = tp + fp
    denom_r = tp + fn
    prec = tp / denom_p if denom_p else 0.0
    rec = tp / denom_r if denom_r else 0.0
    return prec, rec, tp, fp, fn


def _coco_metrics_from_stats(stats: Any) -> dict[str, float]:
    return {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "AR_maxDets_1": float(stats[6]),
        "AR_maxDets_10": float(stats[7]),
        "AR_maxDets_100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }


def _coco_eval_per_category(
    coco_gt: Any,
    coco_dt: Any | None,
    COCOeval: Any,
) -> dict[str, dict[str, Any]]:
    """Run COCOeval once per GT category (``params.catIds = [cid]``)."""
    cat_ids = coco_gt.getCatIds()
    id_to_name = {int(c["id"]): str(c.get("name", c["id"])) for c in coco_gt.loadCats(cat_ids)}
    out: dict[str, dict[str, Any]] = {}
    zeros = _coco_metrics_from_stats([0.0] * 12)

    if coco_dt is None:
        for cid in cat_ids:
            cid_i = int(cid)
            sk = str(cid_i)
            out[sk] = {
                "category_id": cid_i,
                "name": id_to_name.get(cid_i, sk),
                **zeros,
            }
        return out

    for cid in cat_ids:
        cid_i = int(cid)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = [cid_i]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        sk = str(cid_i)
        out[sk] = {
            "category_id": cid_i,
            "name": id_to_name.get(cid_i, sk),
            **_coco_metrics_from_stats(ev.stats),
        }
    return out


def _matched_pr_per_category(
    coco_gt: Any,
    detections: list[dict[str, Any]],
    score_thr: float = 0.25,
) -> dict[str, dict[str, Any]]:
    cat_ids = coco_gt.getCatIds()
    out: dict[str, dict[str, Any]] = {}
    for cid in cat_ids:
        cid_i = int(cid)
        prec, rec, tp, fp, fn = _precision_recall_iou50(
            coco_gt, detections, score_thr=score_thr, category_id=cid_i
        )
        sk = str(cid_i)
        cats = coco_gt.loadCats([cid_i])
        name = str(cats[0].get("name", sk)) if cats else sk
        out[sk] = {
            "category_id": cid_i,
            "name": name,
            "precision_iou50_score025": prec,
            "recall_iou50_score025": rec,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def _system_info() -> dict[str, Any]:
    import platform

    import torch

    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_0"] = torch.cuda.get_device_name(0)
    return info


def _git_rev(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _bench_fps(
    weights: Path,
    image_paths: list[Path],
    device: str | None,
    warmup: int,
    imgsz: int | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    if not image_paths:
        return {
            "fps": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "n_images": 0,
            "note": "No images for benchmark",
        }

    model = YOLO(str(weights))
    kw: dict[str, Any] = {"save": False, "verbose": False}
    if device is not None:
        kw["device"] = device
    if imgsz is not None:
        kw["imgsz"] = int(imgsz)

    n_warm = min(warmup, len(image_paths))
    for p in image_paths[:n_warm]:
        model.predict(source=str(p), **kw)

    to_time = image_paths[n_warm:]
    if not to_time:
        return {
            "fps": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "n_images": 0,
            "warmup": warmup,
            "note": "All images used for warmup; no separate timed set",
        }

    times: list[float] = []
    for p in to_time:
        t0 = time.perf_counter()
        model.predict(source=str(p), **kw)
        times.append(time.perf_counter() - t0)

    mean_s = sum(times) / len(times)
    var = sum((t - mean_s) ** 2 for t in times) / len(times)
    std_s = math.sqrt(var)
    return {
        "fps": len(times) / sum(times) if sum(times) > 0 else None,
        "latency_ms_mean": mean_s * 1000.0,
        "latency_ms_std": std_s * 1000.0,
        "n_images": len(times),
        "warmup": warmup,
    }


def _load_sahi_bench_module() -> Any:
    p = Path(__file__).resolve().parent / "sahi_bench.py"
    spec = importlib.util.spec_from_file_location("sahi_bench", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sahi_params_from_yaml(path: Path, imgsz_override: int | None) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = dict(raw) if isinstance(raw, dict) else {}
    if imgsz_override is not None:
        params["yolo_imgsz"] = int(imgsz_override)
    return params


def main() -> None:
    p = argparse.ArgumentParser(description="COCO bbox eval + FPS (EXP-000).")
    p.add_argument("--gt", type=str, required=True, help="COCO GT JSON (e.g. instances_val.json)")
    p.add_argument(
        "--pred",
        type=str,
        required=True,
        help="COCO detections JSON (list or annotations wrapper)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="experiments/results/test_run_metrics.json",
        help="Output metrics JSON",
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="YOLO .pt for FPS/latency benchmark",
    )
    p.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory of val images for FPS benchmark (first --warmup used only for warmup, rest timed)",
    )
    p.add_argument("--device", type=str, default=None, help="Ultralytics device for benchmark")
    p.add_argument("--warmup", type=int, default=2, help="Warmup inferences")
    p.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="Optional path to saved train config.yaml (metadata only)",
    )
    p.add_argument(
        "--prepare-manifest",
        type=str,
        default=None,
        help="Optional path to prepare_manifest.json (metadata only)",
    )
    p.add_argument(
        "--experiment-id",
        type=str,
        default="EXP-000",
        help="Label stored in output JSON (e.g. EXP-001)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Ultralytics predict imgsz for FPS/latency benchmark (default: model default).",
    )
    p.add_argument(
        "--sahi-config",
        type=str,
        default=None,
        help="If set, FPS/latency uses SAHI sliced inference (YAML; see configs/exp003_sahi.yaml).",
    )
    p.add_argument(
        "--skip-inference-benchmark",
        action="store_true",
        help="Skip Ultralytics FPS/latency benchmark (e.g. SAHI ablation grids).",
    )
    p.add_argument(
        "--inference-benchmark-json",
        type=str,
        default=None,
        help=(
            "If set, load this JSON file (object) and use it as inference_benchmark "
            "instead of timing vanilla Ultralytics predict (e.g. ANTS v1 from bench_ants_v1.py). "
            "Overrides --skip-inference-benchmark when both are set."
        ),
    )
    args = p.parse_args()

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("pycocotools is required (pip install pycocotools).", file=sys.stderr)
        sys.exit(1)

    gt_path = Path(args.gt).expanduser().resolve()
    pred_path = Path(args.pred).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not gt_path.is_file():
        print(f"GT not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    if not pred_path.is_file():
        print(f"Predictions not found: {pred_path}", file=sys.stderr)
        sys.exit(1)
    if not weights_path.is_file():
        print(f"Weights not found: {weights_path}", file=sys.stderr)
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"Images dir not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    sahi_cfg_path: Path | None = None
    if args.sahi_config:
        sahi_cfg_path = Path(args.sahi_config).expanduser().resolve()
        if not sahi_cfg_path.is_file():
            print(f"SAHI config not found: {sahi_cfg_path}", file=sys.stderr)
            sys.exit(1)

    bench_json_path: Path | None = None
    if args.inference_benchmark_json:
        bench_json_path = Path(args.inference_benchmark_json).expanduser().resolve()
        if not bench_json_path.is_file():
            print(f"Inference benchmark JSON not found: {bench_json_path}", file=sys.stderr)
            sys.exit(1)

    coco_gt = COCO(str(gt_path))
    dets = _load_predictions(pred_path)
    if not dets:
        print(
            "Warning: empty predictions list; skipping pycocotools COCOeval "
            "(loadRes is not valid on empty input). coco_eval metrics set to zero.",
            file=sys.stderr,
        )
        stats = [0.0] * 12
        coco_metrics = _coco_metrics_from_stats(stats)
        coco_dt = None
    else:
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

        stats = ev.stats
        coco_metrics = _coco_metrics_from_stats(stats)

    coco_eval_per_category = _coco_eval_per_category(coco_gt, coco_dt, COCOeval)
    matched_pr_per_category = _matched_pr_per_category(coco_gt, dets, score_thr=0.25)

    prec, rec, tp, fp, fn = _precision_recall_iou50(coco_gt, dets, score_thr=0.25)
    matched = {
        "precision_iou50_score025": prec,
        "recall_iou50_score025": rec,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "method": "Greedy match per image, IoU>=0.5, same category, score>=0.25",
    }

    # Image paths present on disk for GT val set
    paths: list[Path] = []
    for im in coco_gt.dataset["images"]:
        fn = im["file_name"]
        cand = images_dir / Path(fn).name
        if cand.is_file():
            paths.append(cand)

    repo_root = Path(__file__).resolve().parents[2]
    if bench_json_path is not None:
        raw_bench = json.loads(bench_json_path.read_text(encoding="utf-8"))
        if not isinstance(raw_bench, dict):
            print(
                f"--inference-benchmark-json must contain a JSON object, got {type(raw_bench)}",
                file=sys.stderr,
            )
            sys.exit(1)
        perf = {
            **raw_bench,
            "loaded_from": path_for_artifact(bench_json_path, repo_root),
        }
    elif args.skip_inference_benchmark:
        perf = {"skipped": True, "note": "--skip-inference-benchmark"}
        if args.imgsz is not None:
            perf["imgsz"] = int(args.imgsz)
    else:
        perf = _bench_fps(
            weights_path, paths, args.device, args.warmup, imgsz=args.imgsz
        )
        if args.imgsz is not None:
            perf = {**perf, "imgsz": int(args.imgsz)}

    payload: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "coco_eval": coco_metrics,
        "coco_eval_stats_raw": [float(x) for x in stats],
        "coco_eval_per_category": coco_eval_per_category,
        "matched_pr": matched,
        "matched_pr_per_category": matched_pr_per_category,
        "inference_benchmark": perf,
        "paths": {
            "gt": path_for_artifact(gt_path, repo_root),
            "predictions": path_for_artifact(pred_path, repo_root),
            "weights": path_for_artifact(weights_path, repo_root),
            "images_dir": path_for_artifact(images_dir, repo_root),
        },
        "system_info": _system_info(),
        "git_rev": _git_rev(repo_root),
    }
    if args.train_config:
        payload["paths"]["train_config"] = path_for_artifact(
            Path(args.train_config), repo_root
        )
    if args.prepare_manifest:
        payload["paths"]["prepare_manifest"] = path_for_artifact(
            Path(args.prepare_manifest), repo_root
        )
    if sahi_cfg_path is not None:
        payload["paths"]["sahi_config"] = path_for_artifact(sahi_cfg_path, repo_root)
    if bench_json_path is not None:
        payload["paths"]["inference_benchmark_json"] = path_for_artifact(
            bench_json_path, repo_root
        )
    if not dets:
        payload["coco_eval_note"] = "Skipped COCOeval (empty predictions); metrics zeroed."

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
