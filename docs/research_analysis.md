# Research analysis (living document)

Narrative synthesis for a future paper. **Update this file after each experiment** with new rows, figures, and takeaways. Raw metrics stay in `experiments/results/*.json`; this file is for interpretation and reporting.

- Procedures: [`experiments.md`](experiments.md), [`reproduction.md`](reproduction.md)
- Compare two metric JSONs: `python scripts/evaluation/compare_metrics.py --baseline … --compare … --out …`

---

## Shared setup (EXP-000 / EXP-001 / EXP-002 / EXP-002b / EXP-003 as currently scripted)

| Item | Value |
|------|--------|
| Model | YOLO26n (`yolo26n.pt`) |
| Training | 1 epoch, batch 4, workers 0 for smoke-style runs; **`imgsz=320`** for EXP-000 / EXP-001; **`imgsz=1280`** for EXP-002; **EXP-002b** sweeps 640–1024 on `test_run` ([`scripts/run_exp002b.sh`](../scripts/run_exp002b.sh), [`configs/train/yolo_exp002b.yaml`](../configs/train/yolo_exp002b.yaml)) |
| Data | COCO128 via `scripts/datasets/download_coco128.py`; **same seed and split** where applicable (`seed` from `prepare_dataset.yaml`; split from [`configs/coco128_exp_split.yaml`](../configs/coco128_exp_split.yaml)). **EXP-002 reuses `datasets/processed/test_run`** (no separate processed dir). |
| Evaluation | EXP-000 vs EXP-001: same val GT when using train-only filter. **EXP-002 / EXP-002b:** identical val COCO JSON and val images to EXP-000; only training/inference resolution changes. **EXP-003:** same val GT; inference uses SAHI slices (no retrain); FPS in metrics JSON reflects per-full-image SAHI cost when `evaluate.py` is run with `--sahi-config`. |
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

## EXP-003 — SAHI sliced inference vs vanilla predict

**Goal:** See whether **tiling** (SAHI) on the fixed val set moves metrics toward **high-resolution** behavior **without** retraining at a larger `imgsz`, and at what **FPS/latency** cost vs plain YOLO predict on the same weights.

**Procedure:** [`scripts/run_exp003.sh`](../scripts/run_exp003.sh) — two SAHI runs (`test_run` weights vs `exp002b_imgsz896` weights), `evaluate.py` with `--sahi-config`, `compare_metrics.py` vs `test_run_metrics.json` and `exp002b_imgsz896_metrics.json`, overlays, then [`experiments/results/exp003_sahi_summary.md`](../experiments/results/exp003_sahi_summary.md) (generated by `write_exp003_sahi_summary.py`).

### Quantitative comparison (recorded run)

Values from **`experiments/results/exp003_sahi_vs_baseline.json`** and **`exp003_sahi_vs_exp002b_896.json`**. Baseline **EXP-000** here is smoke `test_run` (**`imgsz=320`**, not 640 — see `experiments/yolo/test_run/config.yaml`).

**A) SAHI + `test_run` weights vs vanilla EXP-000 (same weights, same val GT)**

| Metric | EXP-000 (vanilla) | EXP-003 SAHI base | Δ (SAHI − vanilla) |
|--------|------------------:|------------------:|-------------------:|
| mAP@[.50:.95] | 0.415 | 0.451 | **+0.035** |
| mAP@.50 | 0.488 | 0.557 | **+0.069** |
| mAP_small | 0.0275 | 0.0275 | **0.000** |
| mAP_medium | 0.169 | 0.267 | **+0.098** |
| mAP_large | 0.766 | 0.765 | −0.002 |
| Precision (matched) | 0.929 | 0.821 | −0.107 |
| Recall (matched) | 0.382 | 0.451 | **+0.069** |
| FPS (`evaluate.py` bench) | 23.9 | 32.3 | +8.5 |
| Latency mean (ms) | 41.9 | 31.0 | −11.0 |

**B) SAHI + 896-trained weights vs vanilla 896 predict (EXP-002b)**

| Metric | Vanilla 896 | EXP-003 SAHI 896 | Δ (SAHI − vanilla) |
|--------|------------:|-----------------:|-------------------:|
| mAP@[.50:.95] | 0.497 | 0.461 | **−0.036** |
| mAP@.50 | 0.640 | 0.639 | ~0 |
| mAP_small | 0.210 | 0.179 | **−0.031** |
| mAP_medium | 0.423 | 0.386 | −0.037 |
| mAP_large | 0.692 | 0.637 | −0.056 |
| Precision (matched) | 0.747 | 0.687 | −0.060 |
| Recall (matched) | 0.608 | 0.559 | −0.049 |
| FPS | 21.3 | 22.0 | +0.7 |
| Latency mean (ms) | 46.9 | 45.4 | −1.5 |

**Provenance:** `evaluation_note` and paths inside the two compare JSONs; raw metrics in `test_run_exp003_sahi_base_metrics.json` / `test_run_exp003_sahi_896_metrics.json`.

### Interpretation (draft for paper / discussion)

On this **1-epoch COCO128 val split**, **SAHI with 320-trained weights** raises **overall mAP** and **mAP_medium** clearly, with **higher matched recall** and **lower matched precision** — consistent with **more detections** (and some false positives) from tiled views. **mAP_small is unchanged** versus vanilla 320 predict, so in this run SAHI did **not** improve the COCO *small* area bucket; the main accuracy lift is in **medium-sized** instances.

**SAHI does not substitute for training at 896:** compared to **vanilla single-image predict at 896** on **896-trained** weights, sliced inference is **worse on mAP@[.5:.95], mAP_small, mAP_medium, mAP_large, and matched P/R**. So tiling at inference **hurts** relative to the best-resolution model here, rather than matching it.

**Throughput note:** The compare JSON reports **higher FPS** for SAHI than for vanilla 320 in (A), and only a small FPS difference in (B). Those numbers come from **`evaluate.py`** benchmarks where vanilla rows use **plain Ultralytics `predict`** and SAHI rows use the **SAHI sliced** path — they are **not guaranteed to be directly comparable** (different preprocessing, slice counts, and timing boundaries). Treat **inference deltas vs baseline** as indicative; for ANTS, re-bench both branches with an identical wall-clock protocol if you need strict speed claims.

**Working conclusion for ANTS-style design:** On this benchmark, **inference-time slicing helps the low-res checkpoint mainly via medium-scale AP and recall**, not **mAP_small**, and **underperforms** the dedicated **896** pipeline when that model is available. Prefer **higher `imgsz` training/infer** when accuracy on small objects and overall mAP matter; consider SAHI (or similar) as a **targeted add-on** only where you must keep a small `imgsz` model and accept different precision/recall trade-offs — and validate **mAP_small** explicitly on your domain.

### Caveats

- Single epoch, COCO128, one SAHI config ([`configs/exp003_sahi.yaml`](../configs/exp003_sahi.yaml)); other slice/overlap settings may change small-object AP.
- Matched P/R and COCO area buckets answer different questions; **mAP_small** can stay flat while overall mAP moves.
- FPS comparison across vanilla vs SAHI benchmark modes should be interpreted cautiously (see above).

---

## EXP-A000 — Ant MOT → YOLO baseline (separate domain)

**Goal:** Single-class **ant** detector on high-res video frames with **MOT** annotations, using the **same** Ultralytics train + `infer_yolo.py` + `evaluate.py` stack as COCO experiments.

**Procedure:** [`scripts/run_ants_prepare.sh`](../scripts/run_ants_prepare.sh) (set `ANTS_DATASET_ROOT`) → `datasets/ants_yolo/` with **temporal** train/val split ([`configs/datasets/ants_mot_prepare.yaml`](../configs/datasets/ants_mot_prepare.yaml)). **Smoke:** [`scripts/run_ants_expA000_smoke.sh`](../scripts/run_ants_expA000_smoke.sh). **Full baseline (canonical bundle):** [`scripts/run_ants_expA000_full.sh`](../scripts/run_ants_expA000_full.sh) → `experiments/yolo/ants_expA000_full/`, [`experiments/results/ants_expA000_full_metrics.json`](../experiments/results/ants_expA000_full_metrics.json), relative stats [`ants_expA000_relative_metrics.json`](../experiments/results/ants_expA000_relative_metrics.json), viz under `experiments/visualizations/ants_expA000_full/`, report [`ants_expA000_full_summary.md`](../experiments/results/ants_expA000_full_summary.md). **Legacy** same recipe, different names: `run_ants_expA000.sh` → `ants_expA000/` + `ants_expA000_metrics.json`.

### Dataset summary (recorded prepare)

From **`datasets/ants_yolo/analysis.json`** (Indoor + Outdoor sequences, temporal split):

| Item | Value |
|------|------:|
| Train / val images | 4261 / 1073 |
| Total annotations (train+val) | 120 612 |
| Val annotations | 25 550 |
| Mean objects / frame (global) | ~22.6 (σ ≈ 15.2) |
| Mean bbox area (px²) | ~5041 |
| Mean bbox “side” √(area) (px) | ~70 |
| Custom bucket counts (side &lt;32 / 32–96 / &gt;96 px) | 0 / 120 612 / 0 |

So in **pixel** space, labeled ants are overwhelmingly in the **32–96 px “side”** band (our `analysis.json` buckets), not the &lt;32 px bucket.

### Quantitative comparison — EXP-A000 **smoke** (1 epoch, `imgsz=640`)

Values from **`experiments/results/ants_expA000_smoke_metrics.json`** (`experiment_id`: EXP-A000-smoke). **Not comparable** to COCO128 mAP (different domain, density, and label statistics).

| Metric | Value |
|--------|------:|
| mAP@[.50:.95] | 0.429 |
| mAP@.50 | 0.711 |
| mAP@.75 | 0.494 |
| mAP_small (COCO area) | **−1** (no GT in bucket) |
| mAP_medium (COCO area) | 0.431 |
| mAP_large (COCO area) | **−1** (no GT in bucket) |
| Precision (IoU≥0.5, score≥0.25, matched) | 0.688 |
| Recall (same) | 0.803 |
| TP / FP / FN (matched) | 20 512 / 9 304 / 5 038 |
| FPS (`evaluate.py`, `imgsz=640`) | ~62.5 |
| Latency mean (ms) | ~16.0 |

**Ultralytics val line** (same smoke run, terminal): Box **P/R/mAP50/mAP50-95** ≈ **0.683 / 0.764 / 0.781 / 0.491** on **1073** images / **25 524** instances — close to but not identical to **pycocotools** on exported preds (different matching / aggregation).

### Quantitative comparison — EXP-A000 **full** (20 epochs, `ants_expA000_full`, `EXP-A000-full`)

Values from **`experiments/results/ants_expA000_full_metrics.json`**. Same val split and pipeline as smoke; **reference baseline** for future ants experiments (EXP-A002b / EXP-A003 / EXP-A004).

| Metric | Value |
|--------|------:|
| mAP@[.50:.95] | **0.636** |
| mAP@.50 | **0.914** |
| mAP@.75 | **0.773** |
| mAP_small (COCO area) | **−1** (unchanged: no GT in bucket) |
| mAP_medium (COCO area) | **0.636** |
| mAP_large (COCO area) | **−1** |
| Precision (IoU≥0.5, score≥0.25, matched) | **0.917** |
| Recall (same) | **0.936** |
| TP / FP / FN (matched) | 23 923 / 2 179 / 1 627 |
| FPS (`evaluate.py`, `imgsz=640`) | ~58.1 |
| Latency mean (ms) | ~17.2 |

**Relative size (GT vs preds, score≥0.25):** [`experiments/results/ants_expA000_relative_metrics.json`](../experiments/results/ants_expA000_relative_metrics.json) — GT mean relative area ≈ **0.00441** (fraction of frame); **26 102** pred boxes above threshold vs **25 550** val GT annotations; pred/GT distributions align in the same ~0.44%–of–frame band (see percentiles in JSON).

### Smoke → full (Δ full − smoke)

| Metric | Smoke | Full | Δ |
|--------|------:|-----:|--:|
| mAP@[.50:.95] | 0.429 | 0.636 | **+0.207** |
| mAP@.50 | 0.711 | 0.914 | **+0.203** |
| mAP_medium | 0.431 | 0.636 | **+0.205** |
| Precision (matched) | 0.688 | 0.917 | **+0.229** |
| Recall (matched) | 0.803 | 0.936 | **+0.133** |
| FP (matched) | 9 304 | 2 179 | **−7 125** |
| FN (matched) | 5 038 | 1 627 | **−3 411** |
| FPS | ~62.5 | ~58.1 | ~−4.4 |
| Latency mean (ms) | ~16.0 | ~17.2 | ~+1.2 |

Twenty epochs deliver a **large** lift in COCO mAP and **much cleaner** greedy matching at IoU 0.25 (far fewer FP, fewer FN). Throughput is **slightly** lower on the full checkpoint (still ~58 FPS on this bench — normal run-to-run variance).

### Training dynamics (`results.png`)

Ultralytics writes **[`experiments/yolo/ants_expA000_full/results.png`](../experiments/yolo/ants_expA000_full/results.png)** (`plots: true` in [`configs/train/yolo.yaml`](../configs/train/yolo.yaml)): train/val **box, cls, DFL** losses decrease; **precision, recall, mAP50, mAP50-95** rise and **plateau** by ~epoch 10–20 with train and val losses **not** diverging sharply — consistent with **stable fitting** rather than obvious late overfit on this split. **Curves are trainer-native** (Ultralytics val); **paper-facing** val numbers should still cite **`evaluate.py`** + pycocotools in `ants_expA000_full_metrics.json`.

**COCO `mAP_small` = −1:** Official COCO buckets use **area in px²** (small &lt; 32², medium up to 96², large above). Your GT boxes sit mainly in **medium** (~5k px²), so **small** and **large** strata are **empty** → pycocotools reports **−1**, not “zero performance.” For “small **in the image**” (fraction of frame), use **`ants_expA000_relative_metrics.json`** (and `analysis.json` pixel buckets).

**Inference log:** Ultralytics warns that **`predict` without `stream=True`** can accumulate results in RAM; val has **1071** benchmarked images — fine here, but for much larger folders consider streaming in [`infer_yolo.py`](../scripts/inference/infer_yolo.py) later.

### Interpretation

**Smoke:** After **one epoch**, the model already reached **decent** mAP@.5 and matched recall, but **many extra boxes** (high FP under greedy matching) — typical for **dense** single-class scenes at default conf/NMS.

**Full:** After **20 epochs**, **mAP@[.5:.95] ~0.64** and **mAP@.5 ~0.91** on val, with matched **P/R both ~0.92**, show a **strong, stable** domain baseline. The remaining **~1.6k FN** and **~2.2k FP** (at score ≥0.25) are the right targets for **resolution sweeps**, **SAHI**, or **post-processing** ablations — compare always to **`ants_expA000_full_metrics.json`**.

### Recommended next steps

1. **EXP-A003 (ants):** SAHI (or similar tiled inference) on val vs vanilla `infer_yolo` at **`imgsz=768`** using weights from **`ants_expA002b_imgsz768`** (or keep comparing to 640 weights if you want a low-res checkpoint). Reference sweep: [`ants_expA002b_resolution_sweep.json`](../experiments/results/ants_expA002b_resolution_sweep.json).
2. **EXP-A004:** ANTS / domain method (to be defined) vs **`ants_expA002b_imgsz768`** or **`ants_expA000_full`** metrics depending on the hypothesis.
3. **Optional:** Revisit **1024** with longer training, different aug, or explicit train/infer policy if the sharp mAP drop is a priority to explain.
4. **Optional:** `stream=True` in [`infer_yolo.py`](../scripts/inference/infer_yolo.py) if val or deployment folders grow very large.

---

## EXP-A002b — Ant resolution sweep (640 / 768 / 896 / 1024)

**Goal:** Quantify how **training + inference resolution** affects **mAP_medium** (primary COCO bucket for this dataset), matched precision/recall, and **FPS/latency** on the fixed ants val split — same pipeline as EXP-A000 full.

**Procedure:** [`scripts/run_ants_expA002b.sh`](../scripts/run_ants_expA002b.sh) — YOLO26n, **20 epochs** per `imgsz` (**640** reused [`ants_expA000_full_metrics.json`](../experiments/results/ants_expA000_full_metrics.json) — identical numbers to EXP-A000 full). Aggregates: [`ants_expA002b_resolution_sweep.json`](../experiments/results/ants_expA002b_resolution_sweep.json), [`ants_expA002b_recommendation.md`](../experiments/results/ants_expA002b_recommendation.md), [`ants_expA002b_relative_metrics.json`](../experiments/results/ants_expA002b_relative_metrics.json), viz `experiments/visualizations/ants_expA002b/`, [`ants_expA002b_summary.md`](../experiments/results/ants_expA002b_summary.md). **Recorded run:** `git_rev` in sweep JSON (e.g. `f0abb00…`); generated UTC in same file.

### Quantitative comparison (recorded run)

| imgsz | mAP@[.50:.95] | mAP@.50 | mAP_medium | P (matched) | R (matched) | FPS | Latency ms |
|------:|--------------:|--------:|-----------:|------------:|------------:|----:|-----------:|
| 640 | 0.636 | 0.914 | 0.636 | 0.917 | 0.936 | 58.1 | 17.2 |
| 768 | **0.645** | **0.922** | **0.645** | 0.914 | **0.946** | **60.6** | **16.5** |
| 896 | 0.626 | 0.919 | 0.628 | 0.921 | 0.943 | 60.0 | 16.7 |
| 1024 | 0.524 | 0.916 | 0.529 | 0.920 | 0.943 | 57.6 | 17.4 |

**Δ vs 640 (same val GT):** 768 → mAP@[.5:.95] **+0.009**, mAP_medium **+0.009**, matched recall **+0.010**, precision **−0.003**; FPS **+2.4**. 896 → mAP@[.5:.95] **−0.009**, mAP_medium **−0.008**. 1024 → mAP@[.5:.95] **−0.112**, mAP_medium **−0.107** (large regression despite similar mAP@.5 and recall).

**Source:** `summary` in [`ants_expA002b_resolution_sweep.json`](../experiments/results/ants_expA002b_resolution_sweep.json); per-run `ants_expA002b_imgsz*_metrics.json`.

**Relative-area preds (score≥0.25):** mean relative bbox area ≈ **0.0045** at 640/768, **0.0047** at 896, **0.0054** at 1024 — see [`ants_expA002b_relative_metrics.json`](../experiments/results/ants_expA002b_relative_metrics.json) (`by_imgsz`).

### Interpretation

- **768 is the clear optimum on this sweep:** best **mAP@[.5:.95]**, **mAP_medium**, **mAP@.5**, **matched recall**, **FPS**, and **lowest latency**. It satisfies the scripted trade-off (FPS ≥ median → max mAP_medium) with **chosen_imgsz = 768** ([`ants_expA002b_recommendation.md`](../experiments/results/ants_expA002b_recommendation.md)).
- **896** sits between 768 and 640 on strict COCO mAP but **below 640** on mAP@[.5:.95] / mAP_medium — not attractive vs 640 or 768 here.
- **1024** shows a **strong drop in mAP@[.5:.95] and mAP_medium** while **mAP@.5** and **matched recall** stay high. That pattern is consistent with **many IoU-strict false positives or poorly localized boxes** (good “hit” at 0.5, poor refinement at higher IoU). Relative pred areas **increase** at 1024, which may reflect **larger predicted boxes** or different error modes; worth checking FP/FN panels under `experiments/visualizations/ants_expA002b/imgsz1024/comparisons/`.
- **Dense scenes:** Matched precision dips slightly at 768 vs 640 while recall rises — more detections, slightly more greedy FP at IoU 0.5 — still a favorable net for mAP on this split.

### Conclusion

For the **ants** val split and **YOLO26n, 20 epochs**, **`imgsz=768`** is the **recommended training/inference resolution**: strictly better **localization / COCO mAP** than 640 and 896 on this run, **faster** than 640/1024 on the `evaluate.py` bench, and **avoid 1024** for this recipe unless you investigate the mAP collapse (train stability, batch/VRAM, or longer schedules).

### Caveats

- Single schedule (20 epochs, batch 4); 1024 might improve with more epochs or tuned aug — not tested here.
- COCO **mAP_small** / **mAP_large** remain **−1** on this GT (same as EXP-A000); narrative uses **mAP_medium** + matched P/R.
- Overlay caps during viz default to 250 images unless `ANTS_VIZ_MAX_IMAGES=all`.

### Next steps (research)

1. **EXP-A003:** Run SAHI (or tiling) at inference on val, anchored to **768-trained** weights, vs vanilla 768 predict — same question as EXP-003 on COCO but ants domain.
2. **Optional ablation:** Short study on **why 1024 underperforms** (learning rate vs image size, box loss, duplicate aug) if deployment truly needs native 1080p tiles.
3. **Jetson / edge:** Re-bench FPS at 768 vs 640 on target hardware; desktop numbers here favor 768 but edge may differ.

---

## Changelog

| Date | Experiment(s) | Summary |
|------|----------------|---------|
| 2026-03-21 | EXP-000 vs EXP-001 | Initial write-up; train-only small-box filter; no mAP_small gain, overall metrics slightly worse on recorded run. |
| 2026-03-21 | EXP-002 (documented) | Pipeline: same `test_run` data, `imgsz` 320→1280; compare JSON includes FPS/latency deltas. |
| 2026-03-21 | EXP-002 (numbers) | Recorded `exp002_vs_baseline.json`: large **mAP_small** / mAP_medium gain; mAP@[.5:.95] and mAP_large down; FPS −5, latency +11 ms. |
| 2026-03-21 | EXP-002b (documented) | Resolution sweep 640–1024; `exp002b_resolution_sweep.json`, recommendation MD, plots under `experiments/results/plots/`. |
| 2026-03-22 | EXP-002b (numbers) | Sweep: best overall **mAP / mAP_small** at **896**; best **mAP_large / FPS** at **640**; **1024** below **896** on mAP and mAP_small; scripted trade-off **768** (FPS ≥ median rule). |
| 2026-03-22 | EXP-003 (documented) | SAHI pipeline: `run_exp003.sh`, `infer_sahi_yolo.py`, `evaluate.py --sahi-config`, compare vs `test_run` and `exp002b_imgsz896` vanilla metrics; summary `exp003_sahi_summary.md`. |
| 2026-03-22 | EXP-003 (numbers) | SAHI+320 weights: mAP +0.035, mAP_medium +0.098, mAP_small unchanged, recall up / precision down vs vanilla 320. SAHI+896 weights: below vanilla 896 on mAP, mAP_small, P/R. |
| 2026-03-22 | EXP-A000 (documented) | Ant MOT→YOLO `prepare_ants_mot.py`, temporal split, EXP-A000 smoke/baseline scripts, `ants_expA000_summary.md` generator; domain split from COCO benchmark. |
| 2026-03-22 | EXP-A000 smoke (numbers) | 1 ep, 640: val mAP@[.5:.95]≈0.429, mAP@.5≈0.711, mAP_medium≈0.431; COCO mAP_small/large −1 (no GT in those area bins); matched P≈0.69, R≈0.80; ~120k boxes, ~23 obj/img; RTX 4070. |
| 2026-03-22 | EXP-A000 full (documented) | `run_ants_expA000_full.sh`: `ants_expA000_full/`, `ants_expA000_full_metrics.json`, `ants_expA000_relative_metrics.json`, viz `ants_expA000_full/`, `ants_expA000_full_summary.md`; training curves `results.png`. |
| 2026-03-22 | EXP-A000 full (numbers) | 20 ep, 640: val mAP@[.5:.95]≈0.636, mAP@.5≈0.914, mAP_medium≈0.636; matched P≈0.92, R≈0.94; FP/FN down sharply vs smoke; FPS ~58; reference baseline for ants A002b/A003/A004. |
| 2026-03-21 | EXP-A002b (documented) | Ants resolution sweep scripted: `run_ants_expA002b.sh`, `yolo_ants_expA002b.yaml`, `summarize_ants_resolution_sweep.py`, `ants_relative_sweep_aggregate.py`, `write_ants_expA002b_summary.py`; 640 reuse from `ants_expA000_full` when applicable. |
| 2026-03-22 | EXP-A002b (numbers) | Sweep: **768** best mAP@[.5:.95] (~0.645), mAP_medium (~0.645), mAP@.5, matched R, FPS (~60.6), lowest latency; **896** below 640/768 on mAP; **1024** large mAP / mAP_medium drop (~0.52 / ~0.53) with high mAP@.5/R; trade-off rule picks **768**; 640 = reused EXP-A000 full. |

*(Append new rows when you re-run and refresh JSONs.)*
