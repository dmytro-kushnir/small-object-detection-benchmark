#!/usr/bin/env python3
"""DETR training entrypoint: forwards to train_rfdetr_ants.py (default EXP-A005 config)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    target = root / "scripts" / "train" / "train_rfdetr_ants.py"
    argv = sys.argv[1:]
    if "--config" not in argv and "-h" not in argv and "--help" not in argv:
        argv = ["--config", str(root / "configs" / "expA005_ants_rfdetr.yaml"), *argv]
    cmd = [sys.executable, str(target), *argv]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
