# `experiments/detr/`

RF-DETR (and other DETR-family) training outputs go here: **`experiments/detr/<run_id>/`** (weights, config snapshot, metrics), mirroring `experiments/yolo/`.

This directory is **gitignored** except this README. Compare models via `scripts/evaluation/evaluate.py` + `compare_metrics.py` on COCO preds from `scripts/inference/infer_detr.py`.
