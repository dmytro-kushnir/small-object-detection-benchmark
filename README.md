# small-object-detection-benchmark

Benchmarking **YOLO26** (Ultralytics) and RF-DETR for small object detection on edge devices (Jetson Nano, RTX 4070).

## Setup

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Run commands from the **repository root** so relative paths in `configs/` resolve correctly (or pass absolute paths via Hydra overrides).

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

- `python scripts/inference/infer_yolo.py --weights … --source … --out predictions.json`
- `python scripts/evaluation/evaluate.py` — mock metrics placeholder until COCO eval is wired

## Docker

```bash
docker build -f docker/Dockerfile -t sod-bench .
```

Use an NVIDIA CUDA base image and install the matching PyTorch build for GPU training.
