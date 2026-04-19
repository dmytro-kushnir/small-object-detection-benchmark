# EXP-002b — Resolution sweep 640–1024 (COCO128)

**Status:** Concluded (recorded sweep + plots).

**Question:** What **mAP vs FPS** trade-off does a mid grid of `imgsz` values give vs the extreme 320/1280 story?

**What we did:** YOLO26n, 1 epoch per size in {640, 768, 896, 1024}, same val GT. Aggregated in `exp002b_resolution_sweep.json` with recommendation markdown and plots.

**Conclusion (recorded):** **896** peaks **mAP**, **mAP_small**, and matched recall on this sweep; **640** is fastest with strongest **mAP_large**. Scripted “median FPS” rule in `exp002b_recommendation.md` picks **768** as a compromise—transparent policy, not a universal optimum.

**Paper role:** Background for **`imgsz` policy** (Table S2: “informs `imgsz` policy”; also contextualizes why Camponotus paired compares use **896** for a different reason than ants **768**—see [`research_context.md`](../research_context.md)).

**Key artifacts:** `experiments/results/exp002b_resolution_sweep.json`, `experiments/results/exp002b_recommendation.md`, `experiments/results/plots/exp002b_*.png`.

**See also:** [`experiments.md`](../experiments.md#exp-002b-resolution-sweep), [`research_analysis.md`](../research_analysis.md#exp-002b--resolution-sweep-6401024).
