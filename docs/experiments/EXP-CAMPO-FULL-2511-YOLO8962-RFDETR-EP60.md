# EXP-CAMPO-FULL-2511-YOLO8962-RFDETR-EP60 — Headline Camponotus compare (track_id–majority)

**Status:** Concluded (2511-image export, paired YOLO26n vs long-train RF-DETR Small).

**Question:** On the **refreshed full CVAT export** with **flattened unique filenames**, how do **YOLO26n @896** and **RF-DETR Small** (long schedule, early stopping) compare under **unified `evaluate.py`**?

**What we did:** Track_id–majority split (`train/1772`, `val/694`, `test/45` in the recorded `analysis.json`). YOLO run dir `camponotus_trackidmajor_full_8962`; RF-DETR `camponotus_rfdetr_trackidmajor_896_ep60_es`. Same conf thresholding and multiclass mode as documented in [`research_analysis.md`](../research_analysis.md).

**Conclusion (recorded):** After longer RF-DETR training on this bundle, **RF-DETR leads on COCO mAP** on **val and test** vs this YOLO run, with **higher trophallaxis-class AP and recall** on test (small **n** on test—high variance). **Throughput** remains **YOLO-favored** (~2× FPS on logged benches). Hugging Face Space is **qualitative**; cite JSON for numbers.

**Paper role:** **Tables 2 and 2a** — main *Camponotus* paired comparison (Table S2).

**Key artifacts:** `experiments/results/camponotus_trackidmajor_full_8962_metrics_{val,test}.json`, `experiments/results/camponotus_rfdetr_trackidmajor_896_ep60_es_metrics_{val,test}.json`, `experiments/results/camponotus_rfdetr_ep60es_vs_yolo8962_{val,test}.json`.

**See also:** [`research_analysis.md`](../research_analysis.md#exp-campo-full-2511-yolo8962-rfdetr-ep60--2511-image-bundle-yolo26n-full_8962--rf-detr-long-train-ep60_es-2026-04), [`cli_commands.md`](../cli_commands.md).
