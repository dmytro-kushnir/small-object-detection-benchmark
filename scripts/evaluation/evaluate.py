#!/usr/bin/env python3
"""Unified evaluation entry point (mock): extend with pycocotools COCOeval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Mock evaluator placeholder.")
    p.add_argument("--gt", type=str, help="COCO GT JSON (unused in mock)")
    p.add_argument("--pred", type=str, help="COCO predictions JSON (unused in mock)")
    p.add_argument("--out", type=str, default="metrics.json", help="Output metrics path")
    args = p.parse_args()

    metrics = {
        "mAP": None,
        "precision": None,
        "recall": None,
        "fps": None,
        "latency_ms": None,
        "note": "Mock evaluator; implement COCO mAP with pycocotools here.",
        "gt": args.gt,
        "pred": args.pred,
    }
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
