#!/usr/bin/env bash
# SAHI hyperparameter grid on ants val (no training). See docs/experiments.md EXP-A003.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec env PYTHONUNBUFFERED=1 python3 "$ROOT/scripts/evaluation/run_ants_expA003_sahi_ablation.py" "$@"
