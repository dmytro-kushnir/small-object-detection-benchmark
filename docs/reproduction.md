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

## What to record for a report

- `git rev-parse HEAD` (also embedded as `git_rev` in metrics JSON where applicable).
- From each metrics JSON, at least: `experiment_id`, `coco_eval`, `system_info`, `git_rev`.

Unified evaluation is implemented in `scripts/evaluation/evaluate.py`; dataset preparation lives in `scripts/datasets/prepare_dataset.py`. See [`docs/experiments.md`](experiments.md) for experiment-specific scripts and [`README.md`](../README.md) for day-to-day commands.

## Make wrappers (optional)

```bash
make reproduce-baseline   # ./scripts/run_smoke_test.sh
make reproduce-exp001     # ./scripts/run_exp001.sh
make reproduce-exp002     # ./scripts/run_exp002.sh
```
