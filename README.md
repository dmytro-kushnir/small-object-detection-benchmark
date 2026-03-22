# small-object-detection-benchmark

Benchmarking **YOLO26** (Ultralytics) and RF-DETR for small object detection on edge devices (Jetson Nano, RTX 4070).

## Setup

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Run commands from the **repository root** so relative paths in `configs/` resolve correctly (or pass absolute paths via Hydra overrides).

For an independent verifier-oriented workflow (environment, smoke test, EXP-001 / EXP-002 / EXP-002b / EXP-003, what to log), see [`docs/reproduction.md`](docs/reproduction.md).

**CODECHECK:** This repo includes a root [`codecheck.yml`](codecheck.yml) manifest aligned with the [CODECHECK community workflow](https://codecheck.org.uk/guide/community-workflow-author.html) (see also [project principles](https://codecheck.org.uk/project/)). Submit via the [CODECHECK register](https://github.com/codecheckers/register/issues/new/choose) when ready.

## Prepare a dataset

Place COCO detection JSON and images under paths in [`configs/prepare_dataset.yaml`](configs/prepare_dataset.yaml), or override on the CLI:

```bash
python scripts/datasets/prepare_dataset.py \
  input.coco_json=path/to/instances.json \
  input.images_dir=path/to/images \
  output_dir=datasets/processed/my_run
```

This writes COCO JSON per split, YOLO `images/` + `labels/`, `dataset.yaml`, and `prepare_manifest.json`. See [`docs/datasets.md`](docs/datasets.md).

## Train YOLO26

Point `data` at the generated `dataset.yaml`, then run:

```bash
python scripts/train/train_yolo.py data=datasets/processed/my_run/dataset.yaml model=yolo26n.pt
```

Artifacts go under `experiments/yolo/<run_id>/` (weights, Ultralytics logs, plus `config.yaml`, `metrics.json`, `system_info.json`). Options live in [`configs/train/yolo.yaml`](configs/train/yolo.yaml).

## Other scripts

- `python scripts/inference/infer_yolo.py --weights … --source … --coco-gt …/instances_val.json --out preds.json` — COCO **list** JSON for `pycocotools` (`image_id` aligned to GT)
- `python scripts/evaluation/evaluate.py --gt … --pred … --weights … --images-dir …/val --out experiments/results/test_run_metrics.json` — COCOeval mAP / AR + P/R + FPS

**EXP-000 baseline:** `./scripts/run_smoke_test.sh` (see [`docs/experiments.md`](docs/experiments.md)).

**EXP-000 figures:** `./scripts/run_visualization.sh` (GT/pred overlays from existing `predictions_val.json`; no retrain).

**EXP-001 (filtered GT):** `./scripts/run_exp001.sh` after EXP-000; see [`docs/experiments.md`](docs/experiments.md).

**EXP-002 (higher `imgsz`):** `./scripts/run_exp002.sh` after EXP-000 (same `test_run` data, `imgsz=1280`); see [`docs/experiments.md`](docs/experiments.md).

**EXP-002b (resolution sweep):** `./scripts/run_exp002b.sh` (640–1024, same `test_run` val GT); see [`docs/experiments.md`](docs/experiments.md).

**EXP-003 (SAHI sliced inference):** `./scripts/run_exp003.sh` after EXP-000 and EXP-002b (optional extended workflow; needs `sahi` from `requirements.txt`); see [`docs/experiments.md`](docs/experiments.md).

**EXP-A000 (ant MOT dataset):** set `ANTS_DATASET_ROOT`, run `./scripts/run_ants_prepare.sh` (writes gitignored `datasets/ants_yolo/`), then `./scripts/run_ants_expA000_full.sh` or `make reproduce-ants-full` for the canonical 20-epoch baseline (optional `./scripts/run_ants_expA000_smoke.sh` for a 1-epoch check); see [`docs/experiments.md`](docs/experiments.md). Legacy output names: `./scripts/run_ants_expA000.sh` (deprecated for new work).

## Docker

```bash
docker build -f docker/Dockerfile -t sod-bench .
```

Use an NVIDIA CUDA base image and install the matching PyTorch build for GPU training.
