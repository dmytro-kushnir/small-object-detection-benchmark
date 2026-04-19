# EXP-CAMPO — Sequence-safe YOLO vs RF-DETR (complementary generalization)

**Status:** Concluded (multiple sub-runs; cite the exact JSON bundle per claim).

**Question:** Under **clip-level (sequence-safe)** splits—where whole source clips stay in one split—how do **YOLO** and **RF-DETR** compare, including **@896** rows?

**What we did:** Distinct from **track_id–majority** headline tables: **different val/test images**. Key IDs in [`research_analysis.md`](../research_analysis.md): **EXP-CAMPO-IDEA1-SEQUENCE-SAFE-FULL-100EP** (YOLO @640), **EXP-CAMPO-IDEA1-SEQUENCE-SAFE-FULL-896** (YOLO @896), **EXP-CAMPO-RFDETR-SEQUENCE-SAFE-FULL** (RF-DETR @640), **EXP-CAMPO-RFDETR-SEQUENCE-SAFE-896** (RF-DETR @896). Compare JSONs: `camponotus_rfdetr_sequence_safe_896_vs_yolo896_{val,test}.json`, `camponotus_rfdetr_sequence_safe_val_vs_yolo.json`, etc.

**Conclusion:** **Sequence-safe** metrics answer **“held-out clips”** generalization; **track_id–majority** answers a **different** split policy (documented cross-split overlap). **Do not rank** sequence-safe vs track-majority mAP as if they were the same benchmark—use cross-split notes only as diagnostics.

**Paper role:** **Table 4** — complementary sequence-held-out story vs headline 2511 track-majority tables.

**Key artifacts (non-exhaustive):** `experiments/results/camponotus_idea1_sequence_safe_full_100ep_metrics_{val,test}.json`, `experiments/results/camponotus_idea1_sequence_safe_full_896_metrics_{val,test}.json`, `experiments/results/camponotus_rfdetr_sequence_safe_896_metrics_{val,test}.json`, `experiments/results/camponotus_rfdetr_sequence_safe_896_vs_yolo896_{val,test}.json`.

**See also:** [`research_analysis.md`](../research_analysis.md#exp-campo-rfdetr-sequence-safe-896--rf-detr-small-idea-1-traininfer-896), “How to read this document” split-policy bullets at file top, [`camponotus_research_roadmap.md`](../camponotus_research_roadmap.md).
