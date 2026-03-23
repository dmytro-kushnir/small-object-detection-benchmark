#!/usr/bin/env python3
"""Train RF-DETR (stub): mirror YOLO layout under experiments/detr/<run_id>/ when implemented."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "RF-DETR training entrypoint (stub). "
            "Target layout: experiments/detr/<run_id>/weights/, config.yaml, metrics.json, system_info.json "
            "(same comparability story as train_yolo.py)."
        )
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/train/rf_detr.yaml",
        help="YAML with dataset paths, model id, hyperparameters (see file for placeholders).",
    )
    args = p.parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    print(
        "train_detr.py is not implemented yet.\n\n"
        f"Read {cfg_path} for intended keys.\n"
        "Wire your upstream RF-DETR training (official repo or pip package), then:\n"
        "  - Save best weights under experiments/detr/<run_id>/weights/\n"
        "  - Dump resolved config + git_rev + system_info like train_yolo.py\n"
        "  - Run infer_detr.py → COCO preds → evaluate.py for apples-to-apples vs YOLO.\n",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
