# Reproduction and independent verification

This repository supports a [CODECHECK](https://codecheck.org.uk/process/)-style workflow: a checker clones the repo, runs explicit steps with documented inputs and outputs, and records enough metadata to compare or audit results.

For the [CODECHECK community workflow](https://codecheck.org.uk/guide/community-workflow-author.html), the repository root provides **`README.md`**, **`LICENSE`** (code + data notes), and **`codecheck.yml`** ([spec](https://codecheck.org.uk/spec/config/1.0/)) listing outputs that the baseline script recreates. When you have a paper DOI or preprint, update `paper.reference` (and authors) in `codecheck.yml`. Submission: [CODECHECK register issues](https://github.com/codecheckers/register/issues/new/choose).

## Environment

- **Python:** 3.10+ recommended.
- **Install:** from the repository root, `pip install -r requirements.txt`.
- **Docker (optional):** `docker build -f docker/Dockerfile -t sod-bench .` — use a CUDA-matched PyTorch build if you need GPU training.

`requirements.txt` uses lower bounds (`>=`) for flexibility. After a successful verifier run on your machine, record the exact stack with `pip freeze > env.freeze.txt` and archive that file alongside your report if you need strict replay.

## Minimum independent check (baseline / EXP-000)

From the repo root:

```bash
./scripts/run_smoke_test.sh
```

**Expect:**

- `experiments/results/test_run_metrics.json` (COCO mAP/AR, FPS, `system_info`, `git_rev`, etc.)
- Artifacts under `experiments/yolo/test_run/` (weights, configs, predictions as produced by the script)

Large or binary directories (`datasets/raw/`, `datasets/processed/`, experiment outputs) are typically **gitignored**; a checker regenerates them locally.

## Extended check (EXP-001)

After the baseline exists (`experiments/results/test_run_metrics.json`):

```bash
./scripts/run_exp001.sh
```

**Expect:**

- `experiments/results/test_run_exp001_metrics.json`
- `experiments/results/exp001_vs_baseline.json` if the baseline metrics file was present
- Corresponding paths under `datasets/processed/test_run_exp001/` and `experiments/yolo/test_run_exp001/`

## Extended check (EXP-002)

After the baseline exists (`experiments/results/test_run_metrics.json`):

```bash
./scripts/run_exp002.sh
```

**Expect:**

- `experiments/results/test_run_exp002_metrics.json`
- `experiments/results/exp002_vs_baseline.json` if the baseline metrics file was present
- Artifacts under `experiments/yolo/test_run_exp002/` (same `datasets/processed/test_run` as EXP-000)

## Extended check (EXP-002b)

Resolution sweep on the same prepared data as EXP-000 (optional; longer than EXP-002):

```bash
./scripts/run_exp002b.sh
```

**Expect:**

- `experiments/results/exp002b_imgsz*_metrics.json` (one per `imgsz`)
- `experiments/results/exp002b_resolution_sweep.json`, `experiments/results/exp002b_recommendation.md`, plots under `experiments/results/plots/`
- YOLO dirs `experiments/yolo/exp002b_imgsz640/` … `exp002b_imgsz1024/` (640 may be copied from baseline metrics only if baseline `test_run` was trained at `imgsz=640`)

## Extended check (EXP-003)

SAHI tiled inference on the **same** val split as EXP-000, compared to vanilla metrics for `test_run` and `exp002b_imgsz896` weights (requires EXP-002b metrics and 896 checkpoint):

```bash
./scripts/run_exp003.sh
```

**Expect:**

- `experiments/yolo/test_run_exp003_sahi_{base,896}/predictions_val.json` and `sahi_config.json`
- `experiments/results/test_run_exp003_sahi_{base,896}_metrics.json`
- `experiments/results/exp003_sahi_vs_baseline.json`, `exp003_sahi_vs_exp002b_896.json`, `exp003_sahi_summary.md`
- Overlays under `experiments/visualizations/test_run_exp003_sahi_{base,896}/`

## What to record for a report

- `git rev-parse HEAD` (also embedded as `git_rev` in metrics JSON where applicable).
- From each metrics JSON, at least: `experiment_id`, `coco_eval`, `system_info`, `git_rev`.

Unified evaluation is implemented in `scripts/evaluation/evaluate.py`; dataset preparation lives in `scripts/datasets/prepare_dataset.py` (COCO workflow) and `scripts/datasets/prepare_ants_mot.py` (MOT → YOLO ants). See [`docs/experiments.md`](experiments.md) for experiment-specific scripts and [`README.md`](../README.md) for day-to-day commands.

## Ant dataset (EXP-A000, optional)

Requires a local **MOT-format** ant dataset; set `ANTS_DATASET_ROOT` to its root, then prepare (**writes gitignored `datasets/ants_yolo/`** — not in the repo; regenerate after clone):

```bash
export ANTS_DATASET_ROOT="/path/to/Ant_dataset"
./scripts/run_ants_prepare.sh
./scripts/run_ants_expA000_smoke.sh   # optional 1-epoch check
./scripts/run_ants_expA000_full.sh    # canonical 20-epoch baseline (or make reproduce-ants-full)
# legacy artifact names only (deprecated): ./scripts/run_ants_expA000.sh
```

## Make wrappers (optional)

```bash
make reproduce-baseline   # ./scripts/run_smoke_test.sh
make reproduce-exp001     # ./scripts/run_exp001.sh
make reproduce-exp002     # ./scripts/run_exp002.sh
make reproduce-exp002b    # ./scripts/run_exp002b.sh
make reproduce-exp003     # ./scripts/run_exp003.sh
make reproduce-ants-smoke # ./scripts/run_ants_expA000_smoke.sh (after prepare + ANTS_DATASET_ROOT)
make reproduce-ants-baseline  # legacy: ./scripts/run_ants_expA000.sh (prefer reproduce-ants-full)
make reproduce-ants-full      # ./scripts/run_ants_expA000_full.sh (after prepare)
```
