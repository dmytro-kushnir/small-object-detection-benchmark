"""SAHI sliced inference helpers + full-image timing (EXP-003)."""

from __future__ import annotations

import inspect
import math
import time
from pathlib import Path
from typing import Any

# Optional get_sliced_prediction kwargs (YAML keys); filtered by installed SAHI signature.
_SLICE_PRED_PASSTHROUGH = frozenset(
    {
        "perform_standard_pred",
        "postprocess_type",
        "postprocess_match_metric",
        "postprocess_match_threshold",
        "postprocess_class_agnostic",
        "merge_buffer_length",
        "auto_slice_resolution",
        "progress_bar",
    }
)

_sliced_pred_param_names: frozenset[str] | None = None


def resolve_sahi_device(device: str | None) -> str:
    """Map CLI/env device to a string SAHI/Ultralytics accept.

    ``None`` or ``auto`` → ``cuda:0`` if ``torch.cuda.is_available()`` else ``cpu``.
    Bare digits (e.g. ``0``) → ``cuda:N`` when CUDA is available (explicit GPU).
    """
    import os

    if device is not None and str(device).strip() != "":
        d = str(device).strip()
    else:
        d = os.environ.get("ANTS_DEVICE") or os.environ.get("SMOKE_DEVICE") or ""
        d = str(d).strip()
    if d.lower() in ("", "auto"):
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    low = d.lower()
    if low == "cpu":
        return "cpu"
    if low.startswith("cuda") or low.startswith("mps"):
        return d
    if d.isdigit():
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda:{int(d)}"
        except ImportError:
            pass
        return "cpu"
    return d


def _get_sliced_prediction_param_names() -> frozenset[str]:
    global _sliced_pred_param_names
    if _sliced_pred_param_names is None:
        from sahi.predict import get_sliced_prediction

        _sliced_pred_param_names = frozenset(
            inspect.signature(get_sliced_prediction).parameters
        )
    return _sliced_pred_param_names


def build_sahi_detection_model(
    weights: Path,
    device: str | None,
    sahi_params: dict[str, Any],
) -> Any:
    from sahi import AutoDetectionModel

    model_type = str(sahi_params.get("model_type", "ultralytics"))
    conf = float(sahi_params.get("confidence_threshold", 0.25))
    dev = resolve_sahi_device(device)
    kw_model: dict[str, Any] = {
        "model_type": model_type,
        "model_path": str(weights),
        "confidence_threshold": conf,
        "device": dev,
    }
    yimgsz = sahi_params.get("yolo_imgsz")
    if yimgsz is not None:
        kw_model["image_size"] = int(yimgsz)
    return AutoDetectionModel.from_pretrained(**kw_model)


def slice_predict_kw(sahi_params: dict[str, Any]) -> dict[str, Any]:
    names = _get_sliced_prediction_param_names()
    kw: dict[str, Any] = {
        "slice_height": int(sahi_params["slice_height"]),
        "slice_width": int(sahi_params["slice_width"]),
        "overlap_height_ratio": float(sahi_params["overlap_height_ratio"]),
        "overlap_width_ratio": float(sahi_params["overlap_width_ratio"]),
    }
    if "verbose" in names:
        kw["verbose"] = int(sahi_params.get("sahi_verbose", 0))

    for key in _SLICE_PRED_PASSTHROUGH:
        if key not in names or key not in sahi_params:
            continue
        val = sahi_params[key]
        if val is None:
            continue
        if key == "postprocess_match_threshold":
            kw[key] = float(val)
        elif key == "merge_buffer_length":
            kw[key] = int(val)
        elif key in (
            "perform_standard_pred",
            "postprocess_class_agnostic",
            "auto_slice_resolution",
            "progress_bar",
        ):
            kw[key] = bool(val)
        else:
            kw[key] = val
    return kw


def run_sahi_sliced_on_path(
    image_path: Path,
    detection_model: Any,
    sahi_params: dict[str, Any],
) -> Any:
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image

    image = read_image(str(image_path))
    return get_sliced_prediction(
        image,
        detection_model,
        **slice_predict_kw(sahi_params),
    )


def predictions_to_coco_dets(result: Any, image_id: int) -> list[dict[str, Any]]:
    """SAHI PredictionResult → COCO detection dicts (xywh absolute)."""
    out: list[dict[str, Any]] = []
    for op in getattr(result, "object_prediction_list", []) or []:
        bbox = op.bbox
        x1, y1 = float(bbox.minx), float(bbox.miny)
        w = float(bbox.maxx - bbox.minx)
        h = float(bbox.maxy - bbox.miny)
        cat = op.category
        cid = int(cat.id) if cat is not None else 0
        score_v = op.score
        sc = float(score_v.value) if score_v is not None else 1.0
        out.append(
            {
                "image_id": image_id,
                "category_id": cid,
                "bbox": [x1, y1, w, h],
                "score": sc,
            }
        )
    return out


def bench_sahi_fps(
    weights: Path,
    image_paths: list[Path],
    device: str | None,
    warmup: int,
    sahi_params: dict[str, Any],
) -> dict[str, Any]:
    """Time SAHI sliced prediction per full image (same as infer path)."""
    try:
        import sahi  # noqa: F401
    except ImportError as e:
        return {
            "fps": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "n_images": 0,
            "note": "sahi not installed (pip install sahi)",
            "backend": "sahi",
            "error": str(e),
        }

    if not image_paths:
        return {
            "fps": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "n_images": 0,
            "note": "No images for benchmark",
            "backend": "sahi",
        }

    detection_model = build_sahi_detection_model(weights, device, sahi_params)
    skw = slice_predict_kw(sahi_params)

    n_warm = min(warmup, len(image_paths))
    for p in image_paths[:n_warm]:
        run_sahi_sliced_on_path(p, detection_model, sahi_params)

    to_time = image_paths[n_warm:]
    if not to_time:
        return {
            "fps": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "n_images": 0,
            "warmup": warmup,
            "note": "All images used for warmup; no separate timed set",
            "backend": "sahi",
        }

    times: list[float] = []
    for p in to_time:
        t0 = time.perf_counter()
        run_sahi_sliced_on_path(p, detection_model, sahi_params)
        times.append(time.perf_counter() - t0)

    mean_s = sum(times) / len(times)
    var = sum((t - mean_s) ** 2 for t in times) / len(times)
    std_s = math.sqrt(var)
    yimgsz = sahi_params.get("yolo_imgsz")
    return {
        "fps": len(times) / sum(times) if sum(times) > 0 else None,
        "latency_ms_mean": mean_s * 1000.0,
        "latency_ms_std": std_s * 1000.0,
        "n_images": len(times),
        "warmup": warmup,
        "backend": "sahi",
        "sahi_params": {
            "model_type": str(sahi_params.get("model_type", "ultralytics")),
            "slice_height": skw["slice_height"],
            "slice_width": skw["slice_width"],
            "overlap_height_ratio": skw["overlap_height_ratio"],
            "overlap_width_ratio": skw["overlap_width_ratio"],
            "confidence_threshold": float(sahi_params.get("confidence_threshold", 0.25)),
            "yolo_imgsz": yimgsz,
            **{k: skw[k] for k in sorted(skw) if k not in {"slice_height", "slice_width", "overlap_height_ratio", "overlap_width_ratio"}},
        },
    }
