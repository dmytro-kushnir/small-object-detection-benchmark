# Research analysis (living document)

Narrative synthesis for a future paper. **Update this file after each experiment** with new rows, figures, and takeaways. Raw metrics stay in `experiments/results/*.json`; this file is for interpretation and reporting.

- Procedures: [`experiments.md`](experiments.md), [`reproduction.md`](reproduction.md)
- Compare two metric JSONs: `python scripts/evaluation/compare_metrics.py --baseline … --compare … --out …`

---

## Shared setup (EXP-000 / EXP-001 / EXP-002 / EXP-002b as currently scripted)

| Item | Value |
|------|--------|
| Model | YOLO26n (`yolo26n.pt`) |
| Training | 1 epoch, batch 4, workers 0 for smoke-style runs; **`imgsz=320`** for EXP-000 / EXP-001; **`imgsz=1280`** for EXP-002; **EXP-002b** sweeps 640–1024 on `test_run` ([`scripts/run_exp002b.sh`](../scripts/run_exp002b.sh), [`configs/train/yolo_exp002b.yaml`](../configs/train/yolo_exp002b.yaml)) |
| Data | COCO128 via `scripts/datasets/download_coco128.py`; **same seed and split** where applicable (`seed` from `prepare_dataset.yaml`; split from [`configs/coco128_exp_split.yaml`](../configs/coco128_exp_split.yaml)). **EXP-002 reuses `datasets/processed/test_run`** (no separate processed dir). |
| Evaluation | EXP-000 vs EXP-001: same val GT when using train-only filter. **EXP-002 / EXP-002b:** identical val COCO JSON and val images to EXP-000; only training/inference resolution changes. |
| Primary metrics | COCO mAP / AR (including **mAP_small**) and **inference FPS/latency** from `scripts/evaluation/evaluate.py`; deltas (including FPS/latency) from `scripts/evaluation/compare_metrics.py` |

---

## EXP-001 — Train-only removal of small bounding boxes

**Hypothesis (informal):** Dropping small objects from training might reduce label noise and help the model focus on larger instances, possibly improving small-object AP on val.

**Intervention:** `configs/exp001_prepare.yaml`: `filter.apply_to: train`, `filter.min_area_px: 1024` (instances with area &lt; 1024 px² removed from train YOLO labels only).

### Quantitative comparison (representative run)

Values below are taken from `experiments/results/exp001_vs_baseline.json` (generated after `./scripts/run_smoke_test.sh` then `./scripts/run_exp001.sh`). **Re-record when you re-run** (paths and numbers will change).

| Metric | EXP-000 (baseline) | EXP-001 | Δ (EXP-001 − EXP-000) |
|--------|-------------------|---------|-------------------------|
| mAP@[.50:.95] | 0.415 | 0.413 | −0.003 |
| mAP@.50 | 0.488 | 0.467 | −0.022 |
| mAP_small | 0.0275 | 0.0275 | 0.000 |
| mAP_medium | 0.169 | 0.176 | +0.007 |
| mAP_large | 0.766 | 0.756 | −0.010 |
| Precision (IoU≥0.5, score≥0.25, matched PR) | 0.929 | 0.900 | −0.029 |
| Recall (same) | 0.382 | 0.353 | −0.029 |

**Recorded provenance (example):** baseline `git_rev` in `experiments/results/test_run_metrics.json` at time of export; see `evaluation_note` inside `exp001_vs_baseline.json` for the fair-comparison reminder.

### Interpretation (draft for paper / discussion)

Under this setup, **removing small objects from the training set does not improve small-object detection on validation** (here, mAP_small is unchanged between runs) **and is associated with lower overall mAP@.50, mAP@[.50:.95], matched precision/recall, and mAP_large**, with a small increase in mAP_medium only. Plausible mechanisms: fewer positive examples for small instances at train time; class/scale statistics shift; with only **one epoch** on a **small subset** (COCO128), effects are indicative only.

**Working conclusion:** Removing small objects from the training set does not improve small-object detection performance in this benchmark and may degrade overall detection accuracy; longer training and larger data are needed before strong claims.

### Caveats to carry into the paper

- Single epoch and tiny dataset → high variance; repeat with seeds / full epochs.
- COCO “small” is area &lt; 32² in the original definition; our filter uses 1024 px² — align wording with the exact threshold in methods.
- Matched P/R is an auxiliary diagnostic; primary story should cite COCO mAP/AR.

---

## EXP-002 — Higher training resolution (same dataset as baseline)

**Hypothesis (informal):** Larger `imgsz` improves feature resolution for small instances, raising **mAP_small** at the cost of **slower inference** (lower FPS, higher latency).

**Intervention:** [`configs/train/yolo_exp002.yaml`](../configs/train/yolo_exp002.yaml): train and evaluate on `datasets/processed/test_run` with **`imgsz=1280`** vs **320** in the smoke baseline.

### Quantitative comparison (recorded run)

Values from **`experiments/results/exp002_vs_baseline.json`** (after `./scripts/run_smoke_test.sh` then `./scripts/run_exp002.sh`). **Re-record when you re-run.**

| Metric | EXP-000 (baseline) | EXP-002 | Δ (EXP-002 − EXP-000) |
|--------|-------------------|---------|-------------------------|
| mAP@[.50:.95] | 0.415 | 0.310 | −0.105 |
| mAP@.50 | 0.488 | 0.487 | −0.001 |
| mAP_small | 0.0275 | 0.166 | **+0.138** |
| mAP_medium | 0.169 | 0.274 | +0.105 |
| mAP_large | 0.766 | 0.404 | −0.362 |
| Precision (IoU≥0.5, score≥0.25, matched PR) | 0.929 | 0.605 | −0.324 |
| Recall (same) | 0.382 | 0.480 | +0.098 |
| FPS (`evaluate.py` benchmark) | 23.9 | 18.9 | −5.0 |
| Latency mean (ms) | 41.9 | 53.0 | +11.1 |

**Provenance:** `git_rev` and `system_info` in `experiments/results/test_run_exp002_metrics.json`; fair-comparison text in `evaluation_note` inside `exp002_vs_baseline.json`.

### Interpretation (draft for paper / discussion)

On this **1-epoch COCO128** setup, **raising train/infer `imgsz` from 320 to 1280 strongly increases COCO mAP on the small and medium area buckets** (mAP_small and mAP_medium up), which matches the intuition that higher resolution helps sub-threshold objects. At the same time, **mAP@[.50:.95] and mAP_large fall sharply**, and **matched precision drops** while **matched recall rises**—consistent with **more detections overall** (including on small objects) but **noisier localization or calibration** on large instances after minimal training. **Inference is ~21% slower** (FPS 23.9 → 18.9) with **~11 ms higher mean latency** per image in this benchmark.

**Working conclusion:** Higher resolution is a plausible lever for **small-object AP** here, but **one epoch is not enough** to claim a net win on full COCO mAP or large objects; longer training, LR/augmentation tuning at high `imgsz`, and repeats across seeds are needed before publication-strength claims.

### Caveats

- Single epoch and tiny dataset → high variance; repeat with seeds and more epochs.
- If `EXP002_BATCH` was reduced for VRAM, state it in methods (breaks strict parity with batch-4 baseline).
- EXP-002 training sets Ultralytics `exist_ok` so re-runs overwrite `experiments/yolo/test_run_exp002/` (no `…2` suffix). `run_visualization.sh` still picks the newest `predictions_val.json` under `test_run_exp001*` / `test_run_exp002*` when `PRED` is unset (helps EXP-001 re-runs and old numbered folders).

---

## EXP-002b — Resolution sweep (640–1024)

**Goal:** Find a better **mAP_small vs speed** trade-off than a single extreme resolution (cf. EXP-002 at 1280 vs smoke at 320).

**Procedure:** [`scripts/run_exp002b.sh`](../scripts/run_exp002b.sh) trains YOLO26n for one epoch at each `imgsz` in {640, 768, 896, 1024} on `datasets/processed/test_run`, runs val inference and `evaluate.py` with matching `--imgsz`. Aggregated metrics: [`experiments/results/exp002b_resolution_sweep.json`](../experiments/results/exp002b_resolution_sweep.json); auto-generated narrative: [`experiments/results/exp002b_recommendation.md`](../experiments/results/exp002b_recommendation.md); plots: [`experiments/results/plots/`](../experiments/results/plots/) (`exp002b_mapsmall_vs_imgsz.png`, `exp002b_map_vs_imgsz.png`, `exp002b_fps_vs_imgsz.png`).

### Quantitative summary (recorded run)

Values from **`experiments/results/exp002b_resolution_sweep.json`** (`summary` array; `git_rev` in file). **Re-record when you re-run.**

| imgsz | mAP@[.50:.95] | mAP@.50 | mAP_small | mAP_medium | mAP_large | P (matched) | R (matched) | FPS | Latency mean (ms) |
|------:|---------------:|--------:|----------:|-----------:|----------:|--------------:|------------:|----:|------------------:|
| 640 | 0.487 | 0.599 | 0.147 | 0.350 | 0.772 | 0.826 | 0.559 | 27.5 | 36.4 |
| 768 | 0.485 | 0.591 | 0.171 | 0.378 | 0.726 | 0.797 | 0.578 | 23.2 | 43.1 |
| 896 | **0.497** | **0.640** | **0.210** | **0.423** | 0.692 | 0.747 | **0.608** | 21.3 | 46.9 |
| 1024 | 0.450 | 0.617 | 0.191 | 0.432 | 0.584 | 0.670 | 0.578 | 21.5 | 46.4 |

**Context vs EXP-000 (320, same val):** The sweep does not retrain at 320; for orientation, the recorded EXP-000 row in this doc has mAP@[.50:.95] ≈ 0.415 and mAP_small ≈ 0.028. Even **640** in EXP-002b already shows much higher **mAP_small** (0.147) and overall mAP (0.487)—expected when comparing a **320** train/infer setup to **640+** on the same tiny 1-epoch schedule (different effective resolution, not a controlled single-variable delta from 320).

### Interpretation (draft for paper / discussion)

On this **1-epoch COCO128** sweep, **896 achieves the best mAP@[.50:.95], mAP@.50, mAP_small, mAP_medium, and matched recall**; **640 retains the strongest mAP_large, matched precision, FPS, and lowest latency**. **1024 underperforms 896** on overall mAP and mAP_small despite similar throughput to 896—consistent with **instability or poor calibration after minimal training** at the largest size, not a monotonic “bigger is better” story.

The bundled rule in **`exp002b_recommendation.md`** picks **768** as the scripted trade-off: among runs with **FPS ≥ median** (~22.4), only **640** and **768** qualify (896 and 1024 sit slightly below median FPS here); between those two, **768 wins on mAP_small** (0.171 vs 0.147). That is a **transparent speed floor**, not a claim that 768 is globally optimal: if **raw small-object AP** matters more than the median-FPS constraint, **896** is the empirical peak in this run; if **large-object AP or latency** dominate, **640** is preferable.

**Working conclusion:** A **mid-high resolution (768–896)** is a reasonable compromise on this setup: large gains in small/medium buckets vs the 320 baseline without jumping all the way to 1280 (EXP-002). **896** looks best for **mAP_small and overall mAP** here; **640** is best for **speed and mAP_large**. Re-run across seeds and epochs before strong publication claims.

### Caveats

- Single epoch, COCO128, one recorded sweep → high variance.
- Median-FPS recommendation rule excludes the fastest **mAP_small** setting (896) when its FPS falls just below median; treat `exp002b_recommendation.md` as one explicit policy, not ground truth.
- Compare EXP-002 (1280) separately: it uses a different anchor than the 640–1024 grid and is not directly interpolated from this table.

---

## Changelog

| Date | Experiment(s) | Summary |
|------|----------------|---------|
| 2026-03-21 | EXP-000 vs EXP-001 | Initial write-up; train-only small-box filter; no mAP_small gain, overall metrics slightly worse on recorded run. |
| 2026-03-21 | EXP-002 (documented) | Pipeline: same `test_run` data, `imgsz` 320→1280; compare JSON includes FPS/latency deltas. |
| 2026-03-21 | EXP-002 (numbers) | Recorded `exp002_vs_baseline.json`: large **mAP_small** / mAP_medium gain; mAP@[.5:.95] and mAP_large down; FPS −5, latency +11 ms. |
| 2026-03-21 | EXP-002b (documented) | Resolution sweep 640–1024; `exp002b_resolution_sweep.json`, recommendation MD, plots under `experiments/results/plots/`. |
| 2026-03-22 | EXP-002b (numbers) | Sweep: best overall **mAP / mAP_small** at **896**; best **mAP_large / FPS** at **640**; **1024** below **896** on mAP and mAP_small; scripted trade-off **768** (FPS ≥ median rule). |

*(Append new rows when you re-run and refresh JSONs.)*
