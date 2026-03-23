# Repository architecture

This document summarizes how the repo stays **simple**, **comparable across models**, and **easy to extend** (e.g. RF-DETR vs YOLO26).

## What works well today

| Layer | Role |
|--------|------|
| **Datasets** | `scripts/datasets/prepare_dataset.py` (+ domain scripts like `prepare_ants_mot.py`) produce COCO-aligned splits and manifests. Paths come from YAML/config, not from code. |
| **Unified evaluation** | `scripts/evaluation/evaluate.py` consumes **COCO GT + COCO-style pred list** → mAP / AR, greedy matched P/R, optional FPS (Ultralytics or injected JSON via `--inference-benchmark-json`). |
| **Cross-run comparison** | `scripts/evaluation/compare_metrics.py` diffs two metrics JSONs from `evaluate.py` — **model-agnostic**. |
| **YOLO train/infer** | Isolated in `scripts/train/train_yolo.py`, `scripts/inference/infer_yolo.py`, `configs/train/yolo*.yaml`; artifacts under `experiments/yolo/<run_id>/`. |
| **Shared pred I/O** | `scripts/inference/coco_pred_common.py` — `file_name` → `image_id` mapping and JSON writer so YOLO, SAHI, and future DETR stay aligned. |

## Extension contract (RF-DETR or any detector)

1. **Training** — Implement `scripts/train/train_detr.py` to write under `experiments/detr/<run_id>/` with the same *kind* of metadata as YOLO (`config.yaml`, `metrics.json`, `system_info.json` where applicable). Use `configs/train/rf_detr.yaml` as the single source of hyperparameters.

2. **Inference** — Implement `scripts/inference/infer_detr.py` to emit a **JSON list** of detections:

   - `image_id` (int, must match COCO GT)
   - `category_id` (int)
   - `bbox`: `[x, y, w, h]` absolute pixels (xywh)
   - `score` (float)

   Use `load_gt_filename_to_image_id()` and `write_coco_predictions_json()` from `coco_pred_common.py`.

3. **Evaluate & compare** — Reuse `evaluate.py` and `compare_metrics.py` with the DETR preds and the **same** `--gt` as the YOLO run. If vanilla Ultralytics FPS is meaningless for DETR, pass `--skip-inference-benchmark` and add a small DETR-specific bench script later (same pattern as `bench_ants_v1.py`).

## Scalability notes

- **Ants / ANTS v1** adds many YOLO-specific scripts under `scripts/`; that is **domain + model** logic. Keep new DETR-specific code similarly namespaced (`infer_detr.py`, optional `scripts/inference/detr_*/`) and **never** fork COCOeval inside DETR scripts — always call `evaluate.py`.
- **`models/`** is intentionally lightweight: pointers and small files only (see `models/README.md`).
- **Hydra** is used for YOLO training; DETR can use the same or plain argparse — consistency of **output paths** matters more than the config loader.

## Simplicity trade-offs

- Duplicated `_load_gt_name_to_id` still exists in some ants-only scripts; the **primary** inference paths (`infer_yolo`, `infer_sahi_yolo`) use `coco_pred_common`. Refactor ants tooling when touching those files.
- Optional dependencies (SAHI, sklearn for DBSCAN) stay commented or in `requirements.txt` to keep the default install lean for COCO+YOLO workflows.
