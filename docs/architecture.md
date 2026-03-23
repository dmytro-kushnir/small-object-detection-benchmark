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

1. **Training (ants EXP-A005)** — [`scripts/train/train_rfdetr_ants.py`](../scripts/train/train_rfdetr_ants.py) + [`configs/expA005_ants_rfdetr.yaml`](../configs/expA005_ants_rfdetr.yaml) → `experiments/rfdetr/ants_expA005/` (`weights/best.pth`, `config.yaml`, `system_info.json`). [`train_detr.py`](../scripts/train/train_detr.py) forwards to this script.

2. **Inference** — [`infer_rfdetr.py`](../scripts/inference/infer_rfdetr.py) emits the same **JSON list** as YOLO:

   - `image_id` (int, must match COCO GT)
   - `category_id` (int; ants use **0**)
   - `bbox`: `[x, y, w, h]` absolute pixels (xywh)
   - `score` (float)

   [`infer_detr.py`](../scripts/inference/infer_detr.py) forwards to `infer_rfdetr.py`.

3. **Throughput** — [`bench_rfdetr.py`](../scripts/evaluation/bench_rfdetr.py) times full-image RF-DETR `predict`; embed in metrics via `evaluate.py --inference-benchmark-json` (same pattern as `bench_ants_v1.py`).

4. **Evaluate & compare** — `evaluate.py` + [`compare_ants_expA005.py`](../scripts/evaluation/compare_ants_expA005.py) vs YOLO768 metrics.

## Scalability notes

- **Ants / ANTS v1** adds many YOLO-specific scripts under `scripts/`; that is **domain + model** logic. RF-DETR ants code lives in `train_rfdetr_ants.py`, `infer_rfdetr.py`, `bench_rfdetr.py` (wrappers `train_detr.py` / `infer_detr.py` forward to these). **Never** fork COCOeval inside model scripts — always call `evaluate.py`.
- **`models/`** is intentionally lightweight: pointers and small files only (see `models/README.md`).
- **Hydra** is used for YOLO training; DETR can use the same or plain argparse — consistency of **output paths** matters more than the config loader.

## Simplicity trade-offs

- Duplicated `_load_gt_name_to_id` still exists in some ants-only scripts; the **primary** inference paths (`infer_yolo`, `infer_sahi_yolo`) use `coco_pred_common`. Refactor ants tooling when touching those files.
- Optional dependencies (SAHI, sklearn for DBSCAN) stay commented or in `requirements.txt` to keep the default install lean for COCO+YOLO workflows.
