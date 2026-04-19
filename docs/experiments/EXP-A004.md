# EXP-A004 — ANTS v1 (dense-region two-stage refinement)

**Status:** Concluded (post-merge **fixed** pipeline; no accuracy rescue vs vanilla 768).

**Question:** Can a **dense ROI → refine** pipeline beat single-pass YOLO @768 or SAHI on the ant val split?

**What we did:** Stage-1 full-frame YOLO @768 → dense ROIs → per-ROI refine → merge; metrics use full-pipeline benchmark JSON. Post-fix bundles: `ants_expA004_fixed_*`.

**Conclusion (recorded):** Parity checks pass (merge round-trip; baseline parity on subset). **mAP does not materially exceed** vanilla **768** or close the gap to SAHI in a way that changes deployment recommendations—treat as a **negative architecture experiment** for this benchmark.

**Paper role:** Optional supplement / methods appendix (“we tried region refinement”); **not** a headline result table for the trophallaxis manuscript.

**Key artifacts:** `experiments/results/ants_expA004_fixed_metrics.json`, `experiments/results/ants_expA004_fixed_vs_baseline.json`.

**See also:** [`experiments.md`](../experiments.md#exp-a004-ants-v1--dense-region-refinement-no-retrain), [`research_analysis.md`](../research_analysis.md#exp-a004--ants-v1-region-aware-refinement).
