# EXP-A003 — Ants SAHI vs vanilla @768 (no retrain)

**Status:** Concluded (main compare + optional 54-config ablation).

**Question:** Does SAHI improve COCO mAP vs **vanilla** inference at **768** using the **same** EXP-A002b checkpoint?

**What we did:** SAHI with `configs/expA003_ants_sahi.yaml`; compare to `ants_expA002b_imgsz768_metrics.json`. Optional ablation grid in `ants_expA003_sahi_ablation.json`.

**Conclusion (recorded):** **Vanilla 768 is preferable** to the default SAHI recipe on mAP and mAP_medium; ablation did not find a tiling setting that beats vanilla on headline AP. SAHI can move **matched FP** at a cost to recall—domain-specific.

**Paper role:** Reinforces **no SAHI** on main ant RF-DETR line (Table S2 background).

**Key artifacts:** `experiments/results/ants_expA003_vs_768.json`, `experiments/results/ants_expA003_sahi_metrics.json`, `experiments/results/ants_expA003_sahi_ablation_summary.md` (if ablation run).

**See also:** [`experiments.md`](../experiments.md#exp-a003-ants-sahi-vs-vanilla-imgsz768-no-retrain), [`research_analysis.md`](../research_analysis.md#exp-a003--ants-sahi-vs-vanilla-imgsz768).
