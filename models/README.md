# `models/`

Reserved for **local assets** that should not live under `experiments/` (e.g. vendored configs, pinned checkpoint manifests, TensorRT engines).

- **Do not** commit large weight files; store them outside git or use download scripts.
- **YOLO** weights are usually referenced by Ultralytics name (`yolo26n.pt`) or a path under `experiments/yolo/<run_id>/weights/`.
- **RF-DETR**: add a short note or symlink policy here when you integrate an upstream repo; keep **training/inference code** under `scripts/train/` and `scripts/inference/` so evaluation stays shared.

Detection outputs for benchmarking must remain **COCO list JSON** compatible with `scripts/evaluation/evaluate.py` (see `docs/architecture.md`).
