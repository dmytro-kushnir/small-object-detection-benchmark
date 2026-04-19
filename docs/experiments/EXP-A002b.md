# EXP-A002b — Ant resolution sweep (640–1024)

**Status:** Concluded (aggregate JSON + recommendation + plots).

**Question:** Which `imgsz` should we pair for **YOLO on dense ants** before comparing to RF-DETR?

**What we did:** Same temporal val split as EXP-A000; YOLO26n **20 epochs** per size; 640 may reuse EXP-A000 full metrics when trained at 640.

**Conclusion (recorded):** On the logged sweep, **768** offers the best balance of **mAP / mAP_medium / FPS** for YOLO26n—**downstream ant compares (EXP-A005/A006) use YOLO @768**, not @896. *Camponotus* uses **896** for a **separate** paired YOLO/RF-DETR line (different dataset and rationale).

**Paper role:** Table S2 “indirect baseline context for Table 1”; anchors YOLO side of EXP-A005.

**Key artifacts:** `experiments/results/ants_expA002b_resolution_sweep.json`, `experiments/results/ants_expA002b_recommendation.md`, `experiments/results/ants_expA002b_imgsz768_metrics.json`.

**See also:** [`experiments.md`](../experiments.md#exp-a002b-ant-resolution-sweep-6401024), [`research_analysis.md`](../research_analysis.md#exp-a002b--ant-resolution-sweep-640--768--896--1024).
