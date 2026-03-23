#!/usr/bin/env python3
"""Compose EXP-A006 throughput from detector benchmark + tracking/smoothing runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detector-bench", type=str, required=True)
    p.add_argument("--tracking-stats", type=str, required=True)
    p.add_argument("--smoothing-stats", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    det = _load(Path(args.detector_bench).expanduser().resolve())
    tr = _load(Path(args.tracking_stats).expanduser().resolve())
    sm = _load(Path(args.smoothing_stats).expanduser().resolve())
    out_path = Path(args.out).expanduser().resolve()

    det_lat = det.get("latency_ms_mean")
    det_fps = det.get("fps")
    n_images = det.get("n_images")

    n_for_extra = int(n_images) if isinstance(n_images, (int, float)) and int(n_images) > 0 else 0
    tr_extra = float(tr.get("elapsed_ms", 0.0)) / n_for_extra if n_for_extra else 0.0
    sm_extra = float(sm.get("elapsed_ms", 0.0)) / n_for_extra if n_for_extra else 0.0

    if det_lat is not None:
        lat_total = float(det_lat) + tr_extra + sm_extra
        fps_total = 1000.0 / lat_total if lat_total > 0 else None
    elif det_fps is not None and n_for_extra:
        det_lat_fallback = 1000.0 / float(det_fps)
        lat_total = det_lat_fallback + tr_extra + sm_extra
        fps_total = 1000.0 / lat_total if lat_total > 0 else None
    else:
        lat_total = None
        fps_total = None

    payload = {
        "backend": "rfdetr+bytetrack+smoothing",
        "detector_benchmark_path": str(Path(args.detector_bench).expanduser().resolve()),
        "tracking_stats_path": str(Path(args.tracking_stats).expanduser().resolve()),
        "smoothing_stats_path": str(Path(args.smoothing_stats).expanduser().resolve()),
        "fps": fps_total,
        "latency_ms_mean": lat_total,
        "n_images": n_images,
        "detector_latency_ms_mean": det_lat,
        "tracking_extra_ms_per_image": tr_extra,
        "smoothing_extra_ms_per_image": sm_extra,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
