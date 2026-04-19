# EXP-A006 — RF-DETR + ByteTrack + temporal smoothing (ants)

**Status:** Concluded (temporal post-processing on top of EXP-A005-optimized detector).

**Question:** Does **tracking + smoothing** improve practical metrics vs single-frame RF-DETR at similar COCO mAP?

**What we did:** ByteTrack on RF-DETR boxes; smoothing (drop short tracks, gap fill, track-mean score); combined benchmark for FPS. **YOLO+tracking** video tooling exists but is **outside** this benchmark protocol.

**Conclusion (recorded):** **mAP@[.5:.95]** nearly unchanged vs optimized detector-only; **matched precision rises** and **FP drops** with **recall slightly down**—useful as a **precision-stability** mode when duplicate boxes matter; adds latency (~3 ms mean in logged run).

**Paper role:** **Table 3** (temporal path vs RF-DETR baseline).

**Key artifacts:** `experiments/results/ants_expA006_vs_baseline.json`, `experiments/results/ants_expA006_tracking_metrics.json`, `experiments/results/ants_expA006_summary.md`, `configs/expA006_ants_tracking.yaml`.

**See also:** [`experiments.md`](../experiments.md#exp-a006-rf-detr--bytetrack--temporal-smoothing-ants), [`research_analysis.md`](../research_analysis.md#exp-a006--rf-detr--bytetrack--temporal-smoothing-ants).
