# EXP-A000 — Ant MOT → YOLO (dense ants domain)

**Status:** Concluded (canonical **full** 20-epoch baseline).

**Question:** What is a strong **YOLO-only** reference on the ant tracking dataset before resolution sweeps, SAHI, and RF-DETR?

**What we did:** MOT → YOLO/COCO prep with **per-sequence temporal split**; train YOLO26n (smoke 1 epoch optional; **full** 20 epochs canonical). Primary COCO story uses **mAP_medium** (mAP_small often −1 on this GT).

**Conclusion:** EXP-A000 **full** is the **reference detector** for EXP-A002b–A006. Relative-area stats and viz support qualitative claims about scale.

**Paper role:** Indirect context for ant tables (YOLO operating point originates here before EXP-A002b picks **768**).

**Key artifacts:** `experiments/results/ants_expA000_full_metrics.json`, `experiments/results/ants_expA000_full_summary.md`, `experiments/yolo/ants_expA000_full/`.

**See also:** [`experiments.md`](../experiments.md#exp-a000-ant-mot--yolo-baseline-domain-dataset), [`research_analysis.md`](../research_analysis.md#exp-a000--ant-mot--yolo-baseline-separate-domain).
