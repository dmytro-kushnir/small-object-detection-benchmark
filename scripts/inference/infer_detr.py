#!/usr/bin/env python3
"""DETR-family inference entrypoint: forwards to infer_rfdetr.py (RF-DETR, EXP-A005)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    target = here / "infer_rfdetr.py"
    cmd = [sys.executable, str(target), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
