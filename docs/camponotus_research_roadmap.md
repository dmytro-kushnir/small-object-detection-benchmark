# Camponotus research roadmap (detection vs interaction)

This document aligns the three research directions with **one canonical annotation** (CVAT → [`prepare_camponotus_detection_dataset.py`](../scripts/datasets/prepare_camponotus_detection_dataset.py)) and **derived exports** for later phases. See also [`docs/camponotus_cvat_workflow.md`](camponotus_cvat_workflow.md).

## Idea 1 — Two-class detection (current benchmark)

**Goal:** Treat trophallaxis as a second detection class on single frames (`ant` vs `trophallaxis`).

**Data:** `datasets/camponotus_yolo` / `datasets/camponotus_coco` (classes 0/1). Map CVAT `state` or two rectangle labels in the prepare script.

**Models:** YOLO26 ([`train_yolo.py`](../scripts/train/train_yolo.py)) and RF-DETR ([`train_rfdetr_ants.py`](../scripts/train/train_rfdetr_ants.py) with [`configs/expCAMPO_rfdetr.yaml`](../configs/expCAMPO_rfdetr.yaml) after [`prepare_camponotus_coco_rfdetr.py`](../scripts/datasets/prepare_camponotus_coco_rfdetr.py)).

**Evaluation:** Unified [`evaluate.py`](../scripts/evaluation/evaluate.py). **Inference** for RF-DETR on two classes uses [`infer_rfdetr.py`](../scripts/inference/infer_rfdetr.py) **`--class-id-mode multiclass`**.

**Orchestration:** [`scripts/run_camponotus_rfdetr_exp.sh`](../scripts/run_camponotus_rfdetr_exp.sh) (train → val/test infer → compare vs YOLO metrics).

## Idea 2 — Interaction modeling (tracking-based)

**Goal:** Detector outputs **one class** (`ant`); trophallaxis is inferred from **spatial–temporal** relations (e.g. proximity + minimum duration between track identities).

**Data (training):** Derived **ant-only** export — [`export_camponotus_ant_only_for_idea2.py`](../scripts/datasets/export_camponotus_ant_only_for_idea2.py) → `datasets/camponotus_coco_ant_only` / `datasets/camponotus_yolo_ant_only`. Each COCO annotation keeps a non-standard boolean **`trophallaxis_gt`** for future event-level evaluation (ignored by standard COCOeval). Preserve **`track_id`** in CVAT exports when practical.

**Data (evaluation, proposed):** Ground-truth **events** (pairs of `track_id`s or boxes with overlapping `trophallaxis_gt` in time) are **not** produced by this repo yet. After Idea 1 metrics, define explicitly:

- pairing rules (IoU / centroid distance thresholds),
- minimum contiguous frames for a positive event,
- how `state` / `trophallaxis_gt` on both ants maps to one interaction label.

**Metrics (future):** Event-centric precision/recall or temporal IoU — separate from bbox mAP.

**Relation to ants EXP-A006:** [`run_ants_expA006.sh`](../scripts/run_ants_expA006.sh) adds tracking + smoothing for **detection** stability, not behavioral interaction detection.

## Idea 3 — Segmentation-assisted interaction

**Status:** **Deferred** until Idea 2 is specified and it is clear that bbox proximity is insufficient.

**Scope:** Instance masks (manual, SAM-assisted, or trained segmenter) to refine contact/overlap; higher annotation or compute cost. No Camponotus segmentation pipeline in this repository yet.

## Annotation hygiene (all phases)

- Keep **`state`** consistent with the trophallaxis pair policy ([`docs/camponotus_labeling_guidelines.md`](camponotus_labeling_guidelines.md)).
- Use stable **`track_id`** per individual within a sequence when possible; run [`validate_camponotus_dataset.py`](../scripts/datasets/validate_camponotus_dataset.py) with **`--strict-track-id`** if you require at most one box per `(image_id, track_id)`.
- Prefer **one** CVAT export as source of truth; avoid maintaining parallel label trees.

## Dataset layout summary

| Artifact | Role |
|----------|------|
| `datasets/camponotus_yolo`, `datasets/camponotus_coco` | Idea 1 (canonical) |
| `datasets/camponotus_rfdetr_coco` | RF-DETR training layout (train/valid) |
| `datasets/camponotus_yolo_ant_only`, `datasets/camponotus_coco_ant_only` | Idea 2 detector train + `trophallaxis_gt` side signal |
