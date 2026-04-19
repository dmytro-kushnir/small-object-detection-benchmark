# EXP-CAMPO-FULL-2511-YOLO-S8962 — YOLO26s capacity check (2511 bundle)

**Status:** Concluded (same GT as headline nano vs RF-DETR run).

**Question:** Does **YOLO26s** beat **YOLO26n** on headline mAP at **896** when training budget is aligned?

**What we did:** `configs/train/yolo_camponotus_trackidmajor_s896.yaml` → `experiments/yolo/camponotus_trackidmajor_full_s8962/`. Compare JSONs vs nano and vs RF-DETR `ep60_es` checkpoint.

**Conclusion (recorded):** On the logged run, **small does not beat nano** on headline **mAP@[.5:.95]** (val/test); use for **capacity / scaling** discussion, not as the primary table winner.

**Paper role:** Mentioned in manuscript text (not a main summary table); supports “wider YOLO is not automatically better here.”

**Key artifacts:** `experiments/results/camponotus_trackidmajor_full_s8962_metrics_{val,test}.json`, `experiments/results/camponotus_yolo_s8962_vs_nano8962_{val,test}.json`, `experiments/results/camponotus_rfdetr_ep60es_vs_yolo_s8962_{val,test}.json`.

**See also:** [`research_analysis.md`](../research_analysis.md#exp-campo-full-2511-yolo-s8962--yolo26s-full_s8962-vs-nano-vs-rf-detr-ep60_es-2026-04).
