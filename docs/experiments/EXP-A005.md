# EXP-A005 — RF-DETR vs YOLO26 on dense ants

**Status:** Concluded (primary architecture comparison on ants val).

**Question:** After matching splits and evaluation, does **RF-DETR** beat **YOLO @768** on detection quality, and at what speed cost?

**What we did:** `datasets/ants_coco/` for RF-DETR; `evaluate.py` with `--inference-benchmark-json`; compare vs `ants_expA002b_imgsz768_metrics.json`. Optimized RF-DETR inference (`optimize_for_inference`) logged separately.

**Conclusion (recorded):** RF-DETR improves **mAP@[.5:.95]**, **mAP@.5**, and **mAP_medium** by ~**0.018** AP with higher matched recall and fewer FN vs YOLO768; YOLO remains **~2× faster** on recorded FPS. Resolution/train schedule may differ—read `evaluation_note` in compare JSONs before attributing effects to backbone alone.

**Paper role:** **Table 1** (ants YOLO vs RF-DETR baseline / compare).

**Key artifacts:** `experiments/results/ants_expA005_rfdetr_vs_yolo.json`, `experiments/results/ants_expA005_optinfer_rfdetr_vs_yolo.json`, `experiments/results/ants_expA005_rfdetr_summary.md`.

**See also:** [`experiments.md`](../experiments.md#exp-a005-rf-detr-baseline-on-ants-vs-yolo26), [`research_analysis.md`](../research_analysis.md#exp-a005--rf-detr-vs-yolo26-ants).
