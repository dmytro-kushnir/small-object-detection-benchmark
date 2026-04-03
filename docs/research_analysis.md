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

1. **EXP-A003 (ants):** **Done:** SAHI underperformed vanilla 768 on mAP and matched P/R ([`ants_expA003_vs_768.json`](../experiments/results/ants_expA003_vs_768.json)); **54-config ablation** ([`ants_expA003_sahi_ablation.json`](../experiments/results/ants_expA003_sahi_ablation.json)) found **no** mAP / mAP_medium win; best settings ≈ **768×768** tiles + higher conf (see below). Optional: Jetson re-bench, or NMS/post-hoc if optimizing matched FP only.
2. **EXP-A004 (ants):** **Post-fix** run ([`ants_expA004_fixed_vs_baseline.json`](../experiments/results/ants_expA004_fixed_vs_baseline.json)) confirms **merge/parity correctness** but **no material mAP rescue** vs vanilla **768** or **SAHI** (tables below). Optional: full-val **`debug_ants_baseline_parity.py`** (no `--max-images`) if you want parity beyond the recorded 50-image check.
3. **EXP-A005 (ants):** **Done** — RF-DETR baseline numbers recorded below.
4. **EXP-A006 (ants temporal):** Run [`run_ants_expA006.sh`](../scripts/run_ants_expA006.sh) for RF-DETR + ByteTrack + temporal smoothing; primary readout [`ants_expA006_vs_baseline.json`](../experiments/results/ants_expA006_vs_baseline.json).
5. **Optional:** Revisit **1024** with longer training, different aug, or explicit train/infer policy if the sharp mAP drop is a priority to explain.
6. **Optional:** `stream=True` in [`infer_yolo.py`](../scripts/inference/infer_yolo.py) if val or deployment folders grow very large.

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

1. **EXP-A003:** **Completed** — SAHI vs vanilla 768 summarized below; viz `experiments/visualizations/ants_expA003_sahi/`.
2. **Optional ablation:** Short study on **why 1024 underperforms** (learning rate vs image size, box loss, duplicate aug) if deployment truly needs native 1080p tiles.
3. **Jetson / edge:** Re-bench FPS at 768 vs 640 on target hardware; desktop numbers here favor 768 but edge may differ.

---

## EXP-A003 — Ants SAHI vs vanilla imgsz=768

**Goal:** Test whether **tiled SAHI inference** on the fixed ants val split improves **COCO mAP / matched P–R** vs **vanilla** `evaluate.py` at **`imgsz=768`**, using the **same** checkpoint as EXP-A002b 768 (**no retrain**).

**Procedure:** [`scripts/run_ants_expA003.sh`](../scripts/run_ants_expA003.sh) — [`infer_sahi_yolo.py`](../scripts/inference/infer_sahi_yolo.py) → `predictions_val.json`; [`evaluate.py`](../scripts/evaluation/evaluate.py) with **`--imgsz 768`** and **`--sahi-config`** ([`configs/expA003_ants_sahi.yaml`](../configs/expA003_ants_sahi.yaml): 512×512 slices, overlap 0.25, `yolo_imgsz` 768, SAHI merge defaults as in config); [`compare_metrics.py`](../scripts/evaluation/compare_metrics.py) vs **`ants_expA002b_imgsz768_metrics.json`** → [`ants_expA003_vs_768.json`](../experiments/results/ants_expA003_vs_768.json); viz `experiments/visualizations/ants_expA003_sahi/`; [`ants_expA003_summary.md`](../experiments/results/ants_expA003_summary.md).

### Quantitative comparison (recorded run)

Values from **`experiments/results/ants_expA003_vs_768.json`** (`baseline_experiment_id`: EXP-A002b-imgsz768; `compare_experiment_id`: EXP-A003-sahi). **Δ = SAHI − vanilla.** `git_rev` in [`ants_expA003_sahi_metrics.json`](../experiments/results/ants_expA003_sahi_metrics.json). **Re-record when you re-run.**

| Metric | Vanilla 768 | SAHI | Δ |
|--------|------------:|-----:|--:|
| mAP@[.50:.95] | 0.645 | 0.601 | **−0.044** |
| mAP@.50 | 0.922 | 0.897 | −0.025 |
| mAP_medium | 0.645 | 0.602 | **−0.043** |
| Precision (IoU≥0.5, score≥0.25, matched) | 0.914 | 0.836 | **−0.078** |
| Recall (matched) | 0.946 | 0.930 | −0.017 |
| TP / FP / FN (matched) | 24 183 / 2 277 / 1 367 | 23 749 / 4 651 / 1 801 | fewer TP, **+2 374 FP**, +434 FN |
| FPS (`evaluate.py` bench) | ~60.6 | ~41.7 | **−18.9** |
| Latency mean (ms) | ~16.5 | ~24.0 | +7.5 |

**Provenance / fair comparison:** `evaluation_note` in `ants_expA003_vs_768.json` — baseline FPS/latency used **Ultralytics `predict`** at imgsz=768; SAHI branch times **sliced inference per full image** via `evaluate.py --sahi-config`. Treat throughput deltas as **indicative** (different code paths), not a single micro-benchmark.

### Interpretation (draft for paper / discussion)

On this **ants val** split, **SAHI did not improve** the already strong **768** checkpoint: **mAP@[.5:.95]** and **mAP_medium** fall by ~0.04 absolute, and **matched precision** drops sharply (~0.08) while **matched recall** dips slightly. The matched **TP/FP/FN** breakdown is consistent with **many extra unmatched predictions** (**FP +2.3k**) and **slightly fewer correct matches** (**TP −434**, **FN +434**) — a pattern that often appears when **tiling + merge** duplicate or fragment boxes in **dense single-class** scenes, or when slice boundaries interact badly with **small–medium** instances that already fill most of the frame at **768** native resize.

**Throughput:** SAHI is **~31% slower** in this bench (~61 → ~42 FPS; ~16.5 → ~24 ms mean), on top of the accuracy regression — so there is **no accuracy–speed trade-off** favoring SAHI here.

**Working conclusion:** For **YOLO26n ants at 768**, **vanilla full-frame inference is preferable** to the default SAHI recipe in EXP-A003. A **54-config ablation** (below) shows **no** tiling setting beats vanilla **mAP@[.5:.95]** or **mAP_medium**; the best grid points use **768×768** slices and **higher** `confidence_threshold`, i.e. they move SAHI closer to full-frame behavior. SAHI remains plausible only if you explicitly optimize **matched FP** at fixed IoU 0.5 (see ablation), not COCO localization.

### SAHI merge ablation (scripted)

To test whether the negative EXP-A003 result is **merge / tiling settings** rather than SAHI in general, run [`scripts/evaluation/run_ants_expA003_sahi_ablation.py`](../scripts/evaluation/run_ants_expA003_sahi_ablation.py) (`make reproduce-ants-expA003-ablation`). It sweeps `perform_standard_pred`, slice size {512, 640, 768}, overlap {0.10, 0.15, 0.25}, and `confidence_threshold` {0.25, 0.35, 0.45} against **`ants_expA002b_imgsz768_metrics.json`**, and writes [`ants_expA003_sahi_ablation.json`](../experiments/results/ants_expA003_sahi_ablation.json) plus [`ants_expA003_sahi_ablation_summary.md`](../experiments/results/ants_expA003_sahi_ablation_summary.md).

#### Ablation results (recorded run, `n_runs` = 54, `early_stopped` = false)

Source: **`ants_expA003_sahi_ablation.json`** / **`ants_expA003_sahi_ablation_summary.md`** (`git_rev` in JSON). **Baseline** = vanilla 768 from **`ants_expA002b_imgsz768_metrics.json`**.

| Question | Answer |
|----------|--------|
| Any SAHI config **>** baseline mAP@[.5:.95] (0.645)? | **No** |
| Any SAHI config **>** baseline mAP_medium (0.645)? | **No** |
| Any config **lower** matched FP than baseline (2277)? | **Yes** (best FP **1897**, **−380**) |

**Best in grid (by mAP@[.5:.95] and mAP_medium, same run):** `perform_standard_pred: true`, **slice 768**, **overlap 0.15**, **`confidence_threshold` 0.35** — mAP@[.5:.95] **0.614** (Δ **≈ −0.031** vs vanilla), mAP_medium **0.616** (Δ **≈ −0.030**), matched P **0.919** / R **0.938**, TP **23 961** / FP **2122** / FN **1589**. This is the strongest SAHI recipe in the sweep but still **below** vanilla on strict COCO AP.

**Lowest matched FP in grid:** slice **768**, overlap **0.25**, **conf 0.45** — FP **1897** (Δ **−380**), mAP_medium **0.615**, mAP@[.5:.95] **0.614**, R **0.933** (TP **23 839**, FN **1711**). Useful if the deployment metric is **greedy matched precision** at IoU 0.5, not mAP.

**Pattern:** All **top-5 mAP_medium** rows in the auto-summary use **`perform_standard_pred: true`** and **slice size 768** (not 512/640), with overlap **0.15–0.25** and conf **0.25–0.45**. Smaller tiles in the grid stayed **further** below vanilla mAP, consistent with seam/merge noise on **dense** ants val at **768** inference scale.

**Verdict:** Tiling hyperparameters **do not overturn** the EXP-A003 conclusion for **COCO mAP / mAP_medium**; they can **reduce matched FP** at the cost of mAP and recall vs vanilla 768. Re-record this subsection if you re-run the grid.

### Caveats

- One SAHI configuration in the main EXP-A003 table; the ablation above covers a small grid.
- Single GPU bench (RTX 4070 in recorded `ants_expA003_sahi_metrics.json`); edge FPS may differ.
- Overlay inspection (`experiments/visualizations/ants_expA003_sahi/`) can localize failure modes (duplicate FPs at slice seams).

---

## EXP-A004 — ANTS v1 (region-aware refinement)

**Hypothesis (informal):** In **dense** frames, a **second YOLO pass** on **padded ROIs** built from **stage-1** detections may improve **localization** (COCO AP) vs a **single** full-frame pass at 768, at the cost of **extra forward passes**.

**Procedure:** [`scripts/run_ants_expA004.sh`](../scripts/run_ants_expA004.sh) — [`infer_ants_v1.py`](../scripts/inference/infer_ants_v1.py) + [`configs/expA004_ants_v1.yaml`](../configs/expA004_ants_v1.yaml); [`bench_ants_v1.py`](../scripts/evaluation/bench_ants_v1.py) times the **full** pipeline; [`evaluate.py`](../scripts/evaluation/evaluate.py) with **`--inference-benchmark-json`** → metrics JSON (default [`ants_expA004_ants_metrics.json`](../experiments/results/ants_expA004_ants_metrics.json); post-merge-fix bundle [`ants_expA004_fixed_metrics.json`](../experiments/results/ants_expA004_fixed_metrics.json)); [`compare_ants_expA004.py`](../scripts/evaluation/compare_ants_expA004.py) → compare JSON (default [`ants_expA004_vs_baseline.json`](../experiments/results/ants_expA004_vs_baseline.json); fixed [`ants_expA004_fixed_vs_baseline.json`](../experiments/results/ants_expA004_fixed_vs_baseline.json)); viz `experiments/visualizations/ants_expA004/`; summary [`ants_expA004_summary.md`](../experiments/results/ants_expA004_summary.md) / [`ants_expA004_fixed_summary.md`](../experiments/results/ants_expA004_fixed_summary.md).

### Quantitative comparison (post–merge-fix run, `git_rev` in [`ants_expA004_fixed_metrics.json`](../experiments/results/ants_expA004_fixed_metrics.json))

Source: [`ants_expA004_fixed_vs_baseline.json`](../experiments/results/ants_expA004_fixed_vs_baseline.json) (`EXP-A004-ANTS-v1-fixed`, RTX 4070, `git_rev` **8e3356a**). The earlier pre-fix ANTS row (~**0.535** mAP@[.5:.95]) differed by only **~0.001** — the large gap vs **768** is **not** explained by the merge bugs alone.

**ANTS v1 vs vanilla imgsz=768** (**Δ = ANTS − 768**).

| Metric | Vanilla 768 | ANTS v1 (fixed) | Δ (ANTS − 768) |
|--------|------------:|----------------:|---------------:|
| mAP@[.50:.95] | 0.645 | 0.536 | **−0.109** |
| mAP@.50 | 0.922 | 0.767 | **−0.155** |
| mAP_medium | 0.645 | 0.539 | **−0.106** |
| Precision (matched, IoU≥0.5, score≥0.25) | 0.914 | 0.773 | **−0.141** |
| Recall (matched) | 0.946 | 0.794 | **−0.153** |
| TP / FP / FN (matched) | 24 183 / 2 277 / 1 367 | 20 286 / 5 957 / 5 264 | fewer TP, **+3 680 FP**, **+3 897 FN** |
| FPS (`bench_ants_v1` vs vanilla `predict`) | ~60.6 | ~28.9 | **−31.7** |
| Latency mean (ms) | ~16.5 | ~34.6 | **+18.1** |

**ANTS v1 vs SAHI** (same compare JSON; **Δ = ANTS − SAHI**).

| Metric | SAHI (EXP-A003) | ANTS v1 (fixed) | Δ (ANTS − SAHI) |
|--------|----------------:|----------------:|----------------:|
| mAP@[.50:.95] | 0.601 | 0.536 | **−0.064** |
| mAP_medium | 0.602 | 0.539 | **−0.063** |
| Precision (matched) | 0.836 | 0.773 | −0.063 |
| Recall (matched) | 0.930 | 0.794 | **−0.136** |
| TP / FP / FN | 23 749 / 4 651 / 1 801 | 20 286 / 5 957 / 5 264 | fewer TP, more FP/FN vs SAHI |
| FPS | ~41.7 | ~28.9 | **−12.8** |
| Latency mean (ms) | ~24.0 | ~34.6 | **+10.6** |

**ROI usage (`rois.json` stats):** mean **~0.41** ROIs per image, mean ROI area **~3.1%** of image area — dense-grid triggers are **sparse** on this val set with the default config.

**Provenance:** `evaluation_note` in **`ants_expA004_fixed_vs_baseline.json`** — throughput paths differ (full ANTS vs single `predict@768` vs SAHI slices).

**Infer artifact (this run):** stage-1 **`predictions_stage1_val.json`** ≈ **26 460** boxes → merged val preds **26 243** detections — net **small** suppression vs stage-1, while matched **FP** vs 768 stays **~+3.7k**, so the dominant issue is **refine / merge behavior on real ROIs** (and possibly **duplicate or shifted boxes**), not a trivial “double NMS wiped everything” artifact.

### Interpretation (this run)

- **Correctness:** **`debug_ants_merge_roundtrip.py`** (1073 imgs) and **`debug_ants_baseline_parity.py`** (50 imgs, dense ROIs off) both reported **0** mismatches — stage-1–only / merge paths align with the intended baselines. **`make reproduce-ants-expA004-fixed`** still yields **~0.536** mAP@[.5:.95] vs **~0.645** for vanilla **768**, i.e. **merge fixes did not materially change** the headline accuracy gap (pre-fix **~0.535** was already close).
- **Accuracy:** ANTS v1 remains **well below** vanilla **768** on **mAP**, **mAP_medium**, and **matched P/R**. The matched breakdown (**TP −3.9k**, **FP +3.7k**, **FN +3.9k** vs 768) still points to **many extra unmatched predictions** and **many missed GT matches**. Versus **SAHI**, ANTS is **lower on mAP** and **much lower on recall**, and **slower** on this bench.
- **Speed:** Full-pipeline FPS **~29** vs **~61** (768) and **~42** (SAHI) — extra forwards without a COCO AP benefit on this config.
- **Dense ROIs:** With **~0.41** ROIs/image on average (default config), refinement is **sparse**; the global metric hit implies **when refinement runs**, it often **hurts** localization or **adds** errors — inspect **`ants_comparisons/`** and ROI viz for concrete failure modes.
- **Working conclusion:** **Keep vanilla 768** for this ants setup unless a **strong config / method** change reverses the gap. ANTS v1 stays a useful **harness** for experiments (tune `count_threshold`, `merge_strategy`, conf thresholds, `refine_imgsz`, ROI policy).

### Parity and merge fixes (implementation)

The regression above was traced to **post-merge NMS on the full stage-1 set** (Ultralytics already NMS’d) and to **`nms_replace_in_roi` dropping in-ROI stage-1 boxes when the refine stage returned no boxes**. Fixes in [`merge.py`](../scripts/inference/ants_v1/merge.py) / [`pipeline.py`](../scripts/inference/ants_v1/pipeline.py): empty `refined` → return stage-1 unchanged; optional `enable_post_merge_nms`; `batched_nms` with `nms_class_agnostic`; explicit `conf` on `model.predict`.

**Post-fix validation (recorded):**

* **Merge round-trip** — [`debug_ants_merge_roundtrip.py`](../scripts/inference/debug_ants_merge_roundtrip.py) on [`ants_expA002b_imgsz768/predictions_val.json`](../experiments/yolo/ants_expA002b_imgsz768/predictions_val.json) + val GT: **1073** images, **mismatches = 0**.
* **ANTS vs baseline parity** — [`debug_ants_baseline_parity.py`](../scripts/inference/debug_ants_baseline_parity.py): **50** images (`--max-images 50`), dense ROIs off, **mismatches = 0** (optional: rerun without `--max-images` for full val).
* **Fixed metrics bundle** — [`ants_expA004_fixed_metrics.json`](../experiments/results/ants_expA004_fixed_metrics.json) + [`ants_expA004_fixed_vs_baseline.json`](../experiments/results/ants_expA004_fixed_vs_baseline.json) from `make reproduce-ants-expA004-fixed` — canonical numbers for the **post–merge-fix** pipeline (tables above).

---

## EXP-A005 — RF-DETR vs YOLO26 (ants)

**Goal:** Same **train/val split** and COCO **category id 0** as YOLO ants; compare **architecture-level** detection on dense ants after fine-tuning [RF-DETR](https://rfdetr.roboflow.com/latest/) (see [`experiments.md`](experiments.md)).

**Procedure:** [`datasets/ants_coco/`](../datasets/ants_coco/README.md) via [`prepare_ants_coco_rfdetr.py`](../scripts/datasets/prepare_ants_coco_rfdetr.py); [`train_rfdetr_ants.py`](../scripts/train/train_rfdetr_ants.py) + [`infer_rfdetr.py`](../scripts/inference/infer_rfdetr.py); [`bench_rfdetr.py`](../scripts/evaluation/bench_rfdetr.py) + [`evaluate.py`](../scripts/evaluation/evaluate.py) **`--inference-benchmark-json`**; [`compare_ants_expA005.py`](../scripts/evaluation/compare_ants_expA005.py) vs [`ants_expA002b_imgsz768_metrics.json`](../experiments/results/ants_expA002b_imgsz768_metrics.json); viz [`viz_ants_expA005_comparisons.py`](../scripts/visualization/viz_ants_expA005_comparisons.py); orchestrator [`run_ants_expA005.sh`](../scripts/run_ants_expA005.sh) / `make reproduce-ants-expA005`.

**Quantitative comparison (recorded, unoptimized inference):** RF-DETR vs YOLO768: mAP@[.5:.95] `0.645→0.663` (Δ `+0.018`), mAP@0.5 `0.922→0.931` (Δ `+0.009`), mAP_medium `0.645→0.664` (Δ `+0.018`); matched precision (IoU0.5, score>=0.25) `0.914→0.923` (Δ `+0.009`), matched recall `0.946→0.962` (Δ `+0.015`); FPS `~60.6→~29.6` (Δ `-31.0`), latency mean `~16.5→~33.8 ms` (Δ `+17.3`). Matched counts: TP `24183→24568` (+385), FP `2277→2057` (-220), FN `1367→982` (-385). See [`ants_expA005_rfdetr_vs_yolo.json`](../experiments/results/ants_expA005_rfdetr_vs_yolo.json) and [`ants_expA005_rfdetr_metrics.json`](../experiments/results/ants_expA005_rfdetr_metrics.json) for `git_rev` and the full tables. Note **train/input resolution** may differ from YOLO **imgsz=768** — see `evaluation_note` in the compare JSON before attributing differences to backbone alone.

**Quantitative comparison (optimized inference; `optimize_for_inference()` enabled):** RF-DETR vs YOLO768: mAP@[.5:.95] `0.645→0.663` (Δ `+0.018`), mAP@0.5 `0.922→0.931` (Δ `+0.009`), mAP_medium `0.645→0.664` (Δ `+0.018`); matched precision `0.914→0.923` (Δ `+0.009`), matched recall `0.946→0.962` (Δ `+0.015`); FPS `~60.6→~33.3` (Δ `-27.2`), latency mean `~16.5→~30.0 ms` (Δ `+13.5`). Relative to unoptimized RF-DETR: FPS `~29.6→~33.3` (+3.8), latency mean `~33.8→~30.0 ms` (-3.8). Matched counts: TP `24183→24567` (+384), FP `2277→2058` (-219), FN `1367→983` (-384). See [`ants_expA005_optinfer_rfdetr_vs_yolo.json`](../experiments/results/ants_expA005_optinfer_rfdetr_vs_yolo.json) and [`ants_expA005_optinfer_rfdetr_metrics.json`](../experiments/results/ants_expA005_optinfer_rfdetr_metrics.json) for `git_rev` and full tables.

**Interpretation (recorded):**

* **Accuracy:** RF-DETR is consistently better than YOLO768 on this ants split for both unoptimized and optimized runs: mAP@[.5:.95] improves by about **+0.018**, mAP@0.5 by about **+0.009**, and matched recall by about **+0.015**. The matched-count pattern is stable across both RF-DETR runs (**~+384 TP**, **~−219 FP**, **~−384 FN** vs YOLO), indicating fewer duplicate/false detections and fewer misses in dense scenes.
* **Latency/throughput trade-off:** YOLO768 remains much faster. RF-DETR unoptimized is **~29.6 FPS / ~33.8 ms**, optimized is **~33.3 FPS / ~30.0 ms**, while YOLO768 is **~60.6 FPS / ~16.5 ms**.
* **Impact of `optimize_for_inference()`:** Optimization changes speed meaningfully (**+3.8 FPS**, **−3.8 ms** vs unoptimized RF-DETR) with negligible metric movement (differences around 1e-5 to 1e-4 scale, likely non-material).
* **Qualitative expectation:** Given TP/FP/FN shifts and higher recall, RF-DETR should show better dense-cluster behavior and fewer duplicate boxes in `side_by_side/` panels, but this comes at a clear runtime cost.

**Conclusion:** If the priority is **best detection quality** on dense ants, choose **RF-DETR with optimized inference** as the default RF-DETR setting. If the priority is **real-time throughput / latency margin**, keep **YOLO768** as the deployment-first baseline. For next iterations, the most useful direction is reducing RF-DETR latency further (export/engine optimization, quantization, or lighter RF-DETR variants) while preserving the observed recall/FP gains.

---

## EXP-A006 — RF-DETR + ByteTrack + temporal smoothing (ants)

**Goal:** Evaluate whether temporal consistency on sequential frames reduces FP, recovers misses, and stabilizes predictions versus single-frame RF-DETR optimized baseline.

**Procedure:** [`run_ants_expA006.sh`](../scripts/run_ants_expA006.sh) orchestrates optimized prediction reuse/rerun, [`track_rfdetr_bytetrack.py`](../scripts/inference/track_rfdetr_bytetrack.py), [`smooth_tracks_expA006.py`](../scripts/inference/smooth_tracks_expA006.py), composed benchmark [`bench_expA006_tracking.py`](../scripts/evaluation/bench_expA006_tracking.py), eval with [`evaluate.py`](../scripts/evaluation/evaluate.py), compare via [`compare_ants_expA006.py`](../scripts/evaluation/compare_ants_expA006.py), viz via [`viz_ants_expA006_tracking.py`](../scripts/visualization/viz_ants_expA006_tracking.py), and report [`write_ants_expA006_summary.py`](../scripts/evaluation/write_ants_expA006_summary.py).

**Smoothing rules:** remove tracks with length `<3`, fill 1-frame gaps by linear interpolation, set score to per-track average confidence.

**Quantitative comparison (A006 vs EXP-A005 optimized baseline):** mAP@[.5:.95] `0.6634→0.6635` (Δ `+0.00014`), mAP@0.5 `0.9309→0.9357` (Δ `+0.00478`), mAP_medium `0.6639→0.6645` (Δ `+0.00053`); matched precision `0.9227→0.9388` (Δ `+0.0161`), matched recall `0.9615→0.9553` (Δ `-0.0063`). Matched counts: TP `24567→24407` (`-160`), FP `2058→1590` (`-468`), FN `983→1143` (`+160`). Throughput: FPS `33.34→30.51` (Δ `-2.83`), latency mean `29.99→32.77 ms` (Δ `+2.78`). See [`ants_expA006_tracking_metrics.json`](../experiments/results/ants_expA006_tracking_metrics.json) and [`ants_expA006_vs_baseline.json`](../experiments/results/ants_expA006_vs_baseline.json).

**Tracking stats:** `401` total tracks, `384` kept after smoothing (`17` removed for short length), average kept track length `~67.4` frames, tracked detections `25907`, smoothed detections `25997`. Benchmark decomposition: detector `~29.99 ms`, tracking overhead `~2.69 ms/img`, smoothing overhead `~0.09 ms/img`. Current run had `segmentation_filter_enabled: false` in tracking stats (as expected for bbox-only predictions) ([`ants_expA006_tracking_stats.json`](../experiments/rfdetr/ants_expA006_tracking_stats.json), [`ants_expA006_smoothing_stats.json`](../experiments/rfdetr/ants_expA006_smoothing_stats.json), [`ants_expA006_tracking_benchmark.json`](../experiments/rfdetr/ants_expA006_tracking_benchmark.json)).

**Interpretation:** For this configuration, temporal modeling mainly improves **precision / FP suppression** (large FP drop), while slightly reducing recall (more FN). COCO mAP@[.5:.95] is effectively unchanged versus the optimized detector-only baseline; mAP@0.5 improves modestly. Runtime impact is moderate (+~2.7 ms mean latency).

**Conclusion:** EXP-A006 is useful as a **precision-stability mode** for dense scenes where duplicate/false detections matter more than maximum recall. It does not provide a large overall mAP lift over optimized RF-DETR baseline, and it adds runtime overhead.

---

## Camponotus (EXP-CAMPO-001) — YOLO26n, ant / trophallaxis (CVAT)

### Prior work — quantitative trophallaxis in *Camponotus* (paper citation)

For **related work / discussion**: Greenwald *et al.* combine **dorsal 2D barcodes** (identity + trajectory) with a **second camera** imaging **fluorescently labeled crop contents** through a transparent floor, enabling per-ant liquid estimates and **per-event** transfer in lab nests. They study ***Camponotus sanctus*** and ***Camponotus fellah*** and show that **trophallaxis duration does not scale linearly with transferred volume**, and that **flow direction can reverse within a single bout**—arguments for **direct or multimodal** interaction characterization when biological *amounts* matter. This repository’s benchmark uses **conventional video** and **bbox / track** labels on ***C. fellah*** (detection mAP and planned event metrics), **without** fluorescence or mass ground truth—**complementary** to that physiology-forward setup, not a replication.

**Reference (manuscript bibliography):** Greenwald, E., Segre, E. & Feinerman, O. Ant trophallactic networks: simultaneous measurement of interaction patterns and food dissemination. *Scientific Reports* **5**, 12496 (2015). <https://doi.org/10.1038/srep12496>

**BibTeX:**

```bibtex
@article{greenwald2015trophallaxis,
  author  = {Greenwald, Efrat and Segre, Enrico and Feinerman, Ofer},
  title   = {Ant trophallactic networks: simultaneous measurement of interaction patterns and food dissemination},
  journal = {Scientific Reports},
  year    = {2015},
  volume  = {5},
  pages   = {12496},
  doi     = {10.1038/srep12496}
}
```

**Goal:** First **end-to-end** benchmark on the **Camponotus fellah** detection dataset: CVAT COCO export → `prepare_camponotus_detection_dataset.py` → YOLO26n train → `infer_yolo.py` → unified `evaluate.py` (COCOeval + matched P/R + FPS).

**Procedure:** Data prep with **`--split-source auto`** (flat `file_name` in CVAT vs `seq_*/` keys in `splits.json`); optional `state` → class 1 (`trophallaxis`). Train: `experiments/yolo/camponotus_yolo26n/` (`config.yaml` in run dir). Val/test predictions: [`camponotus_yolo26n_val_predictions.json`](../experiments/results/camponotus_yolo26n_val_predictions.json), [`camponotus_yolo26n_test_predictions.json`](../experiments/results/camponotus_yolo26n_test_predictions.json). Metrics: [`camponotus_yolo26n_val_metrics.json`](../experiments/results/camponotus_yolo26n_val_metrics.json), [`camponotus_yolo26n_test_metrics.json`](../experiments/results/camponotus_yolo26n_test_metrics.json).

**Dataset summary (this run, from `datasets/camponotus_processed/analysis.json`):** 126 images resolved; **88 train / 19 val / 19 test**; ~**1.6k** boxes; class balance ~**80% ant (0) / 20% trophallaxis (1)**; `split_source: auto` (seed 42, 0.7 / 0.15 / ~0.15).

### Quantitative results (recorded run, `git_rev` in metrics JSON)

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | mAP_small / mAP_medium | mAP_large | P (matched IoU≥0.5, score≥0.25) | R (matched) | TP | FP | FN | FPS (`evaluate.py`) | Latency mean (ms) |
|-------|---------------:|--------:|--------:|:----------------------:|----------:|--------------------------------:|------------:|---:|---:|---:|--------------------:|------------------:|
| **Val** | **0.885** | **0.935** | **0.928** | −1 / −1 (no GT in bin) | **0.885** | **0.895** | **0.960** | 239 | 28 | 10 | **33.8** | **29.6** |
| **Test** | **0.902** | **0.949** | **0.941** | −1 / −1 | **0.902** | **0.908** | **0.956** | 237 | 24 | 11 | **33.1** | **30.2** |

**Hardware / stack (from metrics):** NVIDIA GeForce **RTX 4070**, PyTorch **2.11.0+cu130**, `imgsz` **640** for inference benchmark. **`inference_benchmark.n_images`** is **17** (val and test rows): two fewer than the 19 images per split in `analysis.json` — likely images with **no predictions** or timing subset; treat FPS/latency as **indicative** for this folder.

**COCO area buckets:** `mAP_small` and `mAP_medium` are **−1** because, at **1920×1080**, essentially all GT boxes fall in COCO’s **large** area bin — the reported **mAP_large** equals overall mAP for this dataset. Do not interpret −1 as “failure”; it means **no objects in those bins**.

### Interpretation (draft)

On this **small** dataset, **YOLO26n** reaches **high COCO mAP and matched recall** on both val and test, with **test slightly above val** on mAP (e.g. mAP@[.5:.95] **0.902 vs 0.885**). **Matched precision** is in the **0.89–0.91** range with **FP counts modest** (24–28) versus **FN** (10–11) — consistent with a model that is **confident and fairly complete** on these frames.

**Working conclusion:** The **pipeline is validated** (prep → train → infer → `evaluate.py`). Numbers are **encouraging** for continued annotation and scaling, but **not** yet a claim about **sequence-level generalization**.

### Caveats (carry forward)

- **`--split-source auto`** assigns frames **without** grouping by clip; **train/val/test leakage** across adjacent frames is likely → metrics can be **optimistic** vs a **sequence-held-out** or **`--split-source manifest`** split once `file_name` aligns with `splits.json`.
- **19 val / 19 test images** → high **variance**; one or two sequences dominate the estimate.
- **Ultralytics val** during training reported similar headline mAP; unified **`evaluate.py`** numbers above are the ones to cite for **cross-model** comparison with RF-DETR / other detectors.
- Re-run after **`git_rev`** changes and refresh this table from the JSON paths above.

### EXP-CAMPO-001-V2 — YOLO26n retrain on expanded in-situ set

**What changed:** Added two additional in-situ videos to the Camponotus base dataset and retrained YOLO26n with the same core train settings (`epochs=100`, `imgsz=640`, batch 16) to refresh the baseline.

**Dataset summary (this run, from `datasets/camponotus_processed/analysis.json`):** 280 images resolved; **196 train / 42 val / 42 test**; **3628** boxes; class balance ~**85.1% ant (0) / 14.9% trophallaxis (1)**; `split_source: auto` (seed 42, 0.7 / 0.15 / 0.15).

**Artifacts:**

- run dir: `experiments/yolo/camponotus_yolo26n_v2/`
- predictions: `experiments/results/camponotus_yolo26n_v2_{val,test}_predictions.json`
- metrics: `experiments/results/camponotus_yolo26n_v2_{val,test}_metrics.json`

**Quantitative results (recorded run):**

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | mAP_small / mAP_medium | mAP_large | P (matched IoU≥0.5, score≥0.25) | R (matched) | TP | FP | FN | FPS (`evaluate.py`) | Latency mean (ms) |
|-------|---------------:|--------:|--------:|:----------------------:|----------:|--------------------------------:|------------:|---:|---:|---:|--------------------:|------------------:|
| **Val (V2)** | **0.843** | **0.962** | **0.875** | −1 / 0.233 | **0.844** | **0.908** | **0.961** | 599 | 61 | 24 | **47.7** | **21.0** |
| **Test (V2)** | **0.902** | **0.983** | **0.931** | −1 / 0.000 | **0.902** | **0.920** | **0.981** | 520 | 45 | 10 | **47.3** | **21.2** |

**Interpretation:** Test mAP@[.5:.95] remains essentially unchanged vs the first Camponotus run (~0.902), while mAP@0.5 and matched precision/recall improve. Val mAP@[.5:.95] is lower than the earlier run, but this is on a larger and different split (42 images vs 19), so direct one-number comparison should be treated as a distribution shift rather than regression by itself.

**Important caveat (observed during inference):** `infer_yolo.py` reported skipped files (e.g., 16 extra in val, 13 extra in test) that were present in `datasets/camponotus_yolo/images/{val,test}` but not in the current COCO GT lists. Metrics remain valid because skipped files are ignored, but this indicates stale files in split folders from prior exports. Before future reruns, clear split image/label folders (or regenerate into a clean output root) to avoid noisy warnings and benchmark ambiguity.

### EXP-CAMPO-TRACKIDMAJORITY-SMOKE-001 — YOLO26n (Idea 1, 2-class state)

**Goal:** Smoke-test the full Idea 1 training/eval pipeline on the Camponotus dataset using the `attributes.track_id`-based **majority** split heuristic and globally unique frame basenames (to avoid filename collisions).

**Dataset & split:**
- COCO rewritten with unique basenames: `datasets/camponotus_processed/camponotus_full_instances_default_unique.json` (frame basenames like `seq_camponotus_001_000001.jpg`).
- Split manifest generated from track majority heuristic: `datasets/camponotus_processed/splits_trackid_majority_full_export_unique.json`.
  - images per split: **train=1188 / val=274 / test=176**
  - leakage QA: **40 / 638** `track_id`s overlapped across splits (`scripts/datasets/qa_track_id_overlap_in_splits.py`).

**Training (smoke):** `yolo26n` for `1 epoch` at `imgsz=640` (batch 4, workers 0) using `datasets/camponotus_yolo/camponotus_full_export_unique_trackidmajor/`.

**Artifacts (unified COCO metrics via `evaluate.py`):**
- Val: [`experiments/results/camponotus_idea1_trackidmajor_smoke_metrics_val.json`](../experiments/results/camponotus_idea1_trackidmajor_smoke_metrics_val.json)
- Test: [`experiments/results/camponotus_idea1_trackidmajor_smoke_metrics_test.json`](../experiments/results/camponotus_idea1_trackidmajor_smoke_metrics_test.json)

**Quantitative results (matched P/R @ IoU≥0.5, score≥0.25):**

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | Precision | Recall | TP/FP/FN | FPS | Latency mean (ms) |
|-------|---------------:|--------:|--------:|----------:|-------:|----------:|----:|-------------------:|
| Val | 0.122 | 0.235 | 0.110 | 0.798 | 0.439 | 1204/305/1539 | 35.9 | 27.8 |
| Test | 0.013 | 0.045 | 0.004 | 0.590 | 0.109 | 398/277/3237 | 29.8 | 33.5 |

**Interpretation (draft):** Under this `track_id`-majority split and with a single-epoch smoke schedule, performance drops strongly on the test split. This is consistent with (1) a harder split strategy (more leakage-avoidance than random splitting) and (2) under-training (1 epoch). Next: longer training and/or a different split policy while reusing the unique-basenames fix.

---

### EXP-CAMPO-TRACKIDMAJORITY-SMOKE-002 — YOLO26n (Idea 2, ant-only)

**Goal:** Train the single-class ant detector on the same track_id-majority split using the Idea 2 export (`export_camponotus_ant_only_for_idea2.py`), where `trophallaxis_gt` is stored per-annotation for downstream interaction modeling.

**Training (smoke):** `yolo26n` for `1 epoch` at `imgsz=640` on `datasets/camponotus_yolo_ant_only/camponotus_full_export_unique_trackidmajor_antonly/`.

**Artifacts (unified COCO metrics via `evaluate.py`):**
- Val: [`experiments/results/camponotus_idea2_antonly_smoke_metrics_val.json`](../experiments/results/camponotus_idea2_antonly_smoke_metrics_val.json)
- Test: [`experiments/results/camponotus_idea2_antonly_smoke_metrics_test.json`](../experiments/results/camponotus_idea2_antonly_smoke_metrics_test.json)

**Quantitative results (matched P/R @ IoU≥0.5, score≥0.25):**

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | Precision | Recall | TP/FP/FN | FPS | Latency mean (ms) |
|-------|---------------:|--------:|--------:|----------:|-------:|----------:|----:|-------------------:|
| Val | 0.220 | 0.478 | 0.191 | 0.753 | 0.541 | 1485/486/1258 | 35.4 | 28.2 |
| Test | 0.059 | 0.139 | 0.045 | 0.642 | 0.166 | 605/338/3030 | 40.8 | 24.5 |

**Interpretation (draft):** Ant-only detection is substantially easier than the 2-class state task under the same split, showing higher mAP and recall. This is detection-only smoke; `trophallaxis_gt` is not used by `evaluate.py` (only ant bbox AP is measured here).

---

### EXP-CAMPO-IDEA1-SEQUENCE-SAFE-FULL-100EP — YOLO26n (Idea 1, 2-class state)

**Goal:** Train the full Idea 1 (two-class state) detector on the **sequence-safe** split manifest and evaluate with unified `evaluate.py` on `val` + `test`.

**Dataset & split (sequence-safe, full):**
- YOLO export: `datasets/camponotus_yolo/camponotus_full_export_unique_sequence_safe/dataset.yaml` (imgsz/models use `ant` / `trophallaxis` = 0/1).
- COCO GT splits used for unified eval:
  - `datasets/camponotus_coco/camponotus_full_export_unique_sequence_safe/annotations/instances_val.json`
  - `datasets/camponotus_coco/camponotus_full_export_unique_sequence_safe/annotations/instances_test.json`
- Image counts (from prep analysis):
  - `train=1277 / val=226 / test=135`
- Class balance (bbox instance fraction):
  - `ant=0.9207 / trophallaxis=0.0793`

**Training:** `yolo26n` for `100 epochs` at `imgsz=640` (batch 4, workers 0), run dir `experiments/yolo/camponotus_idea1_sequence_safe_full_100ep/`.

**Artifacts (unified COCO metrics via `evaluate.py`):**
- Val: `experiments/results/camponotus_idea1_sequence_safe_full_100ep_metrics_val.json`
- Test: `experiments/results/camponotus_idea1_sequence_safe_full_100ep_metrics_test.json`

**Quantitative results (unified `evaluate.py`, matched P/R @ IoU≥0.5, score≥0.25):**

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | Precision | Recall | TP/FP/FN | FPS | Latency mean (ms) |
|-------|---------------:|--------:|--------:|----------:|-------:|----------:|----:|-------------------:|
| Val | 0.146 | 0.299 | 0.125 | 0.600 | 0.469 | 2199/1465/2489 | 21.3 | 46.9 |
| Test | 0.240 | 0.518 | 0.223 | 0.743 | 0.790 | 863/298/230 | 30.3 | 33.0 |

**Training dynamics note:** validation `mAP50-95` peaked around **epoch ~19** (after which it flattened/declined), but `patience=50` allowed training to continue to 100 epochs.

**Caveats (area buckets):** unified `evaluate.py` reports `mAP_small = -1` and `mAP_medium = 0.0` for both val/test on this split (these area bins appear effectively empty under this dataset’s COCO area thresholds, so small/medium AP are not informative here).

---

### EXP-CAMPO-IDEA1-SEQUENCE-SAFE-FULL-896 — YOLO26n (Idea 1, resolution ablation @896)

**Goal:** Compare **training + inference at `imgsz=896`** to the recorded **`imgsz=640`, 100-epoch** sequence-safe run on the **same** Idea 1 sequence-safe split (same GT and image roots).

**Training:** `yolo26n` with `epochs=40`, `imgsz=896`, `batch=8`, `workers=4`, `patience=15`, `seed=42` ([`experiments/yolo/camponotus_idea1_sequence_safe_full_896/config.yaml`](../experiments/yolo/camponotus_idea1_sequence_safe_full_896/config.yaml)). Run dir: `experiments/yolo/camponotus_idea1_sequence_safe_full_896/` (Ultralytics early stopping applied; best weights used for eval).

**Inference / eval:** `infer_yolo.py` and `evaluate.py` with **`--imgsz 896`** matching the run; GT under `datasets/camponotus_coco/camponotus_full_export_unique_sequence_safe/annotations/instances_{val,test}.json`; images under `datasets/camponotus_yolo/camponotus_full_export_unique_sequence_safe/images/{val,test}`.

**Artifacts (unified `evaluate.py`):**

- Val: [`experiments/results/camponotus_idea1_sequence_safe_full_896_metrics_val.json`](../experiments/results/camponotus_idea1_sequence_safe_full_896_metrics_val.json)
- Test: [`experiments/results/camponotus_idea1_sequence_safe_full_896_metrics_test.json`](../experiments/results/camponotus_idea1_sequence_safe_full_896_metrics_test.json)
- Predictions: `experiments/results/camponotus_idea1_sequence_safe_full_896_predictions_{val,test}.json`

**896-only quantitative results (matched P/R @ IoU≥0.5, score≥0.25):**

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | Precision | Recall | TP/FP/FN | FPS | Latency mean (ms) |
|-------|---------------:|--------:|--------:|----------:|-------:|----------:|----:|------------------:|
| Val | 0.188 | 0.380 | 0.161 | 0.623 | 0.538 | 2523/1525/2165 | 33.2 | 30.1 |
| Test | 0.167 | 0.353 | 0.161 | 0.645 | 0.705 | 771/424/322 | 41.5 | 24.1 |

**640 (`100ep`) vs 896 — same split, Δ = 896 − 640:**

Reference 640 metrics: [`camponotus_idea1_sequence_safe_full_100ep_metrics_{val,test}.json`](../experiments/results/camponotus_idea1_sequence_safe_full_100ep_metrics_val.json) (train/infer **`imgsz=640`**; 100 epochs, batch 4, workers 0 — see subsection above).

| Split | mAP@[.50:.95] | mAP@.50 | Precision | Recall | FPS | Latency mean (ms) |
|-------|---------------:|--------:|----------:|-------:|----:|------------------:|
| **Val** 640 | 0.146 | 0.299 | 0.600 | 0.469 | 21.3 | 46.9 |
| **Val** 896 | 0.188 | 0.380 | 0.623 | 0.538 | 33.2 | 30.1 |
| **Δ val** | **+0.042** | **+0.081** | +0.023 | +0.069 | +11.9 | −16.8 |
| **Test** 640 | 0.240 | 0.518 | 0.743 | 0.790 | 30.3 | 33.0 |
| **Test** 896 | 0.167 | 0.353 | 0.645 | 0.705 | 41.5 | 24.1 |
| **Δ test** | **−0.073** | **−0.165** | −0.098 | −0.085 | +11.2 | −8.9 |

**Bundled deltas (`compare_metrics.py`, compare − baseline = 896 − 640):** Full numeric payloads (including exact `mAP_*` diffs and inference deltas) are in:

- Val: [`experiments/results/camponotus_sequence_safe_896_vs_640_100ep_val.json`](../experiments/results/camponotus_sequence_safe_896_vs_640_100ep_val.json)
- Test: [`experiments/results/camponotus_sequence_safe_896_vs_640_100ep_test.json`](../experiments/results/camponotus_sequence_safe_896_vs_640_100ep_test.json)

**Recorded val summary (matches terminal / JSON):** mAP@[.5:.95] **+0.0425**, mAP@0.5 **+0.0806**, mAP_large **+0.0425**, P/R (IoU≥0.5, matched) **+0.023 / +0.069**, FPS **+11.9**, latency mean **−16.7 ms**. **Recorded test summary:** mAP@[.5:.95] **−0.0726**, mAP@0.5 **−0.1654**, P/R **−0.098 / −0.084**, FPS **+11.3**, latency mean **−9.0 ms**.

**Reading `compare_metrics` area buckets on this split:** Both runs report **`mAP_small = −1`** and **`mAP_medium = 0.0`** (COCO small/medium strata effectively unused). The script’s printed **`mAP_small` / `mAP_medium` deltas of 0.000000** are **not** meaningful “no change in small-object AP”; they come from subtracting identical sentinel / zero bucket values. Use **overall mAP** and **`mAP_large`** (≈ overall here) plus matched P/R for interpretation.

**Regenerate compare JSONs:**

```bash
python3 scripts/evaluation/compare_metrics.py \
  --baseline experiments/results/camponotus_idea1_sequence_safe_full_100ep_metrics_val.json \
  --compare experiments/results/camponotus_idea1_sequence_safe_full_896_metrics_val.json \
  --out experiments/results/camponotus_sequence_safe_896_vs_640_100ep_val.json \
  --evaluation-note "Sequence-safe Idea 1; baseline YOLO 640/100ep (b4w0) vs compare 896/40ep early-stop (b8w4). Not single-variable resolution — schedules differ."

python3 scripts/evaluation/compare_metrics.py \
  --baseline experiments/results/camponotus_idea1_sequence_safe_full_100ep_metrics_test.json \
  --compare experiments/results/camponotus_idea1_sequence_safe_full_896_metrics_test.json \
  --out experiments/results/camponotus_sequence_safe_896_vs_640_100ep_test.json \
  --evaluation-note "Sequence-safe Idea 1; baseline YOLO 640/100ep (b4w0) vs compare 896/40ep early-stop (b8w4). Not single-variable resolution — schedules differ."
```

**Interpretation (draft):** On **val**, **896** improves **COCO mAP** (both @.50 and @[.50:.95]) and **matched recall** versus **640/100ep**, with **higher FPS** and **lower latency** in this `evaluate.py` benchmark (RTX 4070). On **test**, the same **896** checkpoint is **worse** on mAP and matched P/R than **640/100ep** — a **train/val vs test gap** (same GT for both models on each split). Causes may include **different training schedules** (100ep @640 b4w0 vs 40ep @896 b8w4 + early stop), **early stopping keyed to val**, or **test split difficulty** relative to val. Treat **test** as the stricter generalization readout; **do not** assume val gains transfer without confirming on test.

**RF-DETR compare note:** Existing Camponotus RF-DETR vs YOLO tables use **`camponotus_idea1_sequence_safe_full_100ep`** as the YOLO reference (**640**). A fair RF-DETR vs **896** comparison would require re-running `compare_camponotus_rfdetr_vs_yolo.py` (or equivalent) against the **896** metrics JSONs.

**Optional quick peek (`jq`, same numbers as rows above):**

```bash
jq -s '
  {
    "640_val": .[0] | {mAP50: .coco_eval.mAP_50, mAP50_95: .coco_eval.mAP_50_95, P: .matched_pr.precision_iou50_score025, R: .matched_pr.recall_iou50_score025, fps: .inference_benchmark.fps},
    "896_val": .[1] | {mAP50: .coco_eval.mAP_50, mAP50_95: .coco_eval.mAP_50_95, P: .matched_pr.precision_iou50_score025, R: .matched_pr.recall_iou50_score025, fps: .inference_benchmark.fps}
  }
' experiments/results/camponotus_idea1_sequence_safe_full_100ep_metrics_val.json \
  experiments/results/camponotus_idea1_sequence_safe_full_896_metrics_val.json
```

(Swap `_val` → `_test` for test.)

---

### EXP-CAMPO-IDEA1-TRACKIDMAJORITY-FULL-40EP-B8W4 — YOLO26n (Idea 1, 2-class state)

**Goal:** Re-run Idea 1 on the harder `track_id`-majority split with a stronger/faster schedule than smoke (`batch=8`, `workers=4`, early stopping) to test if the identity-oriented split remains tractable with longer training.

**Dataset & split (track_id-majority, full):**
- YOLO export: `datasets/camponotus_yolo/camponotus_full_export_unique_trackidmajor/dataset.yaml`
- COCO GT splits:
  - `datasets/camponotus_coco/camponotus_full_export_unique_trackidmajor/annotations/instances_val.json`
  - `datasets/camponotus_coco/camponotus_full_export_unique_trackidmajor/annotations/instances_test.json`
- Split leakage QA from manifest generation: `40 / 638` overlapping `track_id`s (`datasets/camponotus_processed/trackid_overlap_qa_full_export_unique.json`).

**Training run:**
- Config: `epochs=40`, `imgsz=640`, `batch=8`, `workers=4`, `patience=15`
- Actual behavior: early-stopped after 35 epochs; best checkpoint at epoch 20
- Run dir: `experiments/yolo/camponotus_idea1_trackidmajor_full_40ep_b8w4/`

**Artifacts (unified COCO metrics via `evaluate.py`):**
- Val: `experiments/results/camponotus_idea1_trackidmajor_full_40ep_b8w4_metrics_val.json`
- Test: `experiments/results/camponotus_idea1_trackidmajor_full_40ep_b8w4_metrics_test.json`

**Quantitative results (matched P/R @ IoU≥0.5, score≥0.25):**

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | Precision | Recall | TP/FP/FN | FPS | Latency mean (ms) |
|-------|---------------:|--------:|--------:|----------:|-------:|----------:|----:|-------------------:|
| Val | 0.365 | 0.638 | 0.359 | 0.815 | 0.767 | 2105/477/638 | 37.2 | 26.8 |
| Test | 0.293 | 0.525 | 0.329 | 0.664 | 0.394 | 1431/724/2204 | 31.0 | 32.3 |

**Comparison vs prior track_id-majority smoke (1 epoch, same split policy):**
- **Val:** mAP@[.5:.95] `0.122 → 0.365` (**+0.243**), mAP@.50 `0.235 → 0.638` (**+0.403**), matched P/R `0.798/0.439 → 0.815/0.767`.
- **Test:** mAP@[.5:.95] `0.013 → 0.293` (**+0.280**), mAP@.50 `0.045 → 0.525` (**+0.480**), matched P/R `0.590/0.109 → 0.664/0.394`.
- **Interpretation:** The earlier “near-failure” smoke behavior on this split appears to be mostly under-training; with a moderate schedule and larger batch, the same split becomes substantially more usable.

**Comparison note vs sequence-safe full run (`EXP-CAMPO-IDEA1-SEQUENCE-SAFE-FULL-100EP`):**
- On **val**, this track_id-majority run is much stronger (`mAP@[.5:.95] 0.365 vs 0.146`, `mAP@.50 0.638 vs 0.299`) and faster benchmark throughput.
- On **test**, `mAP@[.5:.95]` is also higher (`0.293 vs 0.240`), but matched recall is lower (`0.394 vs 0.790`).
- Because split policies differ (sequence-held-out vs majority-by-identity heuristic with residual overlap), these runs are **not apples-to-apples** for final ranking; treat them as complementary stress tests of different generalization assumptions.

---

### EXP-CAMPO-IDEA1-TRACKIDMAJORITY-FULL-896 — YOLO26n (Idea 1, resolution ablation @896)

**Goal:** Same **`track_id`-majority** Idea 1 split as **`EXP-CAMPO-IDEA1-TRACKIDMAJORITY-FULL-40EP-B8W4`**, but **train + infer at `imgsz=896`** (mirrors the sequence-safe 896 recipe) to measure **640 vs 896** under a **matched schedule** (`epochs=40`, `batch=8`, `workers=4`, `patience=15`).

**Training:** [`experiments/yolo/camponotus_idea1_trackidmajor_full_896/config.yaml`](../experiments/yolo/camponotus_idea1_trackidmajor_full_896/config.yaml). **Observed behavior:** early stopping after **35** epochs; **best checkpoint at epoch 20** (same pattern as the 640 `b8w4` run). Post-train Ultralytics val line on **274** images (trainer-native, not pycocotools): all-class **P/R/mAP50/mAP50-95 ≈ 0.743 / 0.741 / 0.769 / 0.396** — **paper-facing** numbers remain **`evaluate.py`** below.

**Artifacts (unified `evaluate.py`):**

- Val: [`experiments/results/camponotus_idea1_trackidmajor_full_896_metrics_val.json`](../experiments/results/camponotus_idea1_trackidmajor_full_896_metrics_val.json)
- Test: [`experiments/results/camponotus_idea1_trackidmajor_full_896_metrics_test.json`](../experiments/results/camponotus_idea1_trackidmajor_full_896_metrics_test.json)
- Predictions: `experiments/results/camponotus_idea1_trackidmajor_full_896_predictions_{val,test}.json`

**896-only quantitative results (matched P/R @ IoU≥0.5, score≥0.25):**

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | Precision | Recall | TP/FP/FN | FPS | Latency mean (ms) |
|-------|---------------:|--------:|--------:|----------:|-------:|----------:|----:|------------------:|
| Val | 0.319 | 0.637 | 0.276 | 0.768 | 0.805 | 2208/668/535 | 53.4 | 18.7 |
| Test | 0.293 | 0.536 | 0.306 | 0.691 | 0.491 | 1785/799/1850 | 41.5 | 24.1 |

**640 (`40ep` b8w4) vs 896 — same split, Δ = 896 − 640:**

| Split | mAP@[.50:.95] | mAP@.50 | Precision | Recall | FPS | Latency mean (ms) |
|-------|---------------:|--------:|----------:|-------:|----:|------------------:|
| **Val** 640 | 0.365 | 0.638 | 0.815 | 0.767 | 37.2 | 26.8 |
| **Val** 896 | 0.319 | 0.637 | 0.768 | 0.805 | 53.4 | 18.7 |
| **Δ val** | **−0.047** | **~0** | −0.048 | **+0.038** | **+16.2** | **−8.1** |
| **Test** 640 | 0.293 | 0.525 | 0.664 | 0.394 | 31.0 | 32.3 |
| **Test** 896 | 0.293 | 0.536 | 0.691 | 0.491 | 41.5 | 24.1 |
| **Δ test** | **−0.0008** | **+0.012** | +0.027 | **+0.097** | **+10.5** | **−8.2** |

**Bundled deltas (`compare_metrics.py`, compare − baseline = 896 − 640):**

- Val: [`experiments/results/camponotus_trackidmajor_896_vs_640_val.json`](../experiments/results/camponotus_trackidmajor_896_vs_640_val.json)
- Test: [`experiments/results/camponotus_trackidmajor_896_vs_640_test.json`](../experiments/results/camponotus_trackidmajor_896_vs_640_test.json)

**Regenerate compare JSONs:**

```bash
python3 scripts/evaluation/compare_metrics.py \
  --baseline experiments/results/camponotus_idea1_trackidmajor_full_40ep_b8w4_metrics_val.json \
  --compare experiments/results/camponotus_idea1_trackidmajor_full_896_metrics_val.json \
  --out experiments/results/camponotus_trackidmajor_896_vs_640_val.json \
  --evaluation-note "Same track_id-majority split; baseline 640/40ep b8w4 vs compare 896. Fair if same GT and comparable schedule."

python3 scripts/evaluation/compare_metrics.py \
  --baseline experiments/results/camponotus_idea1_trackidmajor_full_40ep_b8w4_metrics_test.json \
  --compare experiments/results/camponotus_idea1_trackidmajor_full_896_metrics_test.json \
  --out experiments/results/camponotus_trackidmajor_896_vs_640_test.json \
  --evaluation-note "Same track_id-majority split; baseline 640/40ep b8w4 vs compare 896. Fair if same GT and comparable schedule."
```

**Interpretation (draft):** With **matched early-stop schedule** to the 640 run, **896 does not improve strict COCO mAP@[.5:.95] on val** (it drops ~**0.047**); **mAP@.50** is effectively **tied** on val. **Matched recall rises** on val (**+0.038**) at the cost of **precision** (**−0.048**) — more greedy matches, not a clean strict-AP win. On **test**, **mAP@[.5:.95]** is **unchanged** within rounding (~**−0.0008**), while **mAP@.50**, **matched P/R**, and **throughput** favor **896** (higher recall, faster `evaluate.py` bench). So **896 is not a universal upgrade** for this split: **val localization (IoU-strict AP)** prefers **640** here; **test** looks more favorable for **896** on **recall / speed** at similar overall AP. **Area buckets:** val retains a small **mAP_medium** signal (~**0.084**); test **mAP_medium = 0** in both runs — interpret medium/large deltas cautiously.

**Qualitative / video:** Example BoT-SORT overlays (`--conf 0.35`): `experiments/visualizations/camponotus_idea1_trackidmajor_full_896_tracked_camponotus_006.mp4`, `…_015.mp4` (not scored; for inspection only).

---

#### Cross-split diagnostic — YOLO896 sequence-safe vs YOLO896 track_id–majority

**Not comparable for ranking:** **val** and **test** here are **different image sets** (sequence-held-out vs `track_id`-majority). Deltas are **diagnostic only** (relative difficulty / distribution shift), not “trackid training beats sequence-safe” on a shared benchmark.

**Procedure:** `compare_metrics.py` with **baseline** = sequence-safe 896 metrics, **compare** = track_id–majority 896 metrics → **Δ = compare − baseline** on the **same split label** (val or test) only.

**Bundles:**

- Val: [`experiments/results/camponotus_crosssplit_seqsafe896_vs_trackid896_val.json`](../experiments/results/camponotus_crosssplit_seqsafe896_vs_trackid896_val.json)
- Test: [`experiments/results/camponotus_crosssplit_seqsafe896_vs_trackid896_test.json`](../experiments/results/camponotus_crosssplit_seqsafe896_vs_trackid896_test.json)

**Recorded Δ (track_id–majority 896 − sequence-safe 896):** **Val:** mAP@[.5:.95] **+0.130**, mAP@0.5 **+0.257**, **mAP_medium** term **+0.084** (sequence-safe val has **mAP_medium = 0**; trackid-majority val does not — not “small-object Δ = 0” semantics), matched P/R **+0.144 / +0.267**, FPS **+20.2**, latency **−11.4 ms**. **Test:** mAP@[.5:.95] **+0.126**, mAP@0.5 **+0.184**, matched P **+0.046**, R **−0.214** (sequence-safe **test** has higher greedy matched recall on **its** GT despite lower mAP — incomparable sets), FPS/latency **~flat** (Δ under **0.03** FPS / **0.02 ms**).

**Regenerate:**

```bash
python3 scripts/evaluation/compare_metrics.py \
  --baseline experiments/results/camponotus_idea1_sequence_safe_full_896_metrics_val.json \
  --compare experiments/results/camponotus_idea1_trackidmajor_full_896_metrics_val.json \
  --out experiments/results/camponotus_crosssplit_seqsafe896_vs_trackid896_val.json \
  --evaluation-note "Different splits — val sets are not the same images. Diagnostic only."

python3 scripts/evaluation/compare_metrics.py \
  --baseline experiments/results/camponotus_idea1_sequence_safe_full_896_metrics_test.json \
  --compare experiments/results/camponotus_idea1_trackidmajor_full_896_metrics_test.json \
  --out experiments/results/camponotus_crosssplit_seqsafe896_vs_trackid896_test.json \
  --evaluation-note "Different splits — test sets are not the same images. Diagnostic only."
```

---

### RF-DETR comparison (EXP-CAMPO-RFDETR)

**Goal:** Same **splits** and **two-class COCO** as YOLO Camponotus; compare RF-DETR vs YOLO26 using unified **`evaluate.py`**.

**Procedure:**

1. Canonical prep: `prepare_camponotus_detection_dataset.py` (and validate) as for EXP-CAMPO-001.
2. Roboflow layout: `python3 scripts/datasets/prepare_camponotus_coco_rfdetr.py` (config [`configs/datasets/camponotus_coco_rfdetr.yaml`](../configs/datasets/camponotus_coco_rfdetr.yaml)).
3. Orchestrated run: [`scripts/run_camponotus_rfdetr_exp.sh`](../scripts/run_camponotus_rfdetr_exp.sh) (uses [`configs/expCAMPO_rfdetr.yaml`](../configs/expCAMPO_rfdetr.yaml); **`infer_rfdetr.py --class-id-mode multiclass`**). Set **`EXP_CAMPO_SKIP_TRAIN=1`** to reuse existing weights under `experiments/rfdetr/camponotus_rfdetr/weights/best.pth`.
4. Metrics: `experiments/results/camponotus_rfdetr_sequence_safe_val_metrics.json`, `camponotus_rfdetr_sequence_safe_test_metrics.json`. Compare JSONs: `camponotus_rfdetr_sequence_safe_val_vs_yolo.json`, `camponotus_rfdetr_sequence_safe_test_vs_yolo.json` (vs sequence-safe YOLO references in `experiments/results/camponotus_idea1_sequence_safe_full_100ep_metrics_{val,test}.json`).

**Quantitative results (recorded run, sequence-safe split):**

- Metrics JSONs:
  - `experiments/results/camponotus_rfdetr_sequence_safe_val_metrics.json`
  - `experiments/results/camponotus_rfdetr_sequence_safe_test_metrics.json`
- Compare vs YOLO (`EXP-CAMPO-IDEA1-SEQUENCE-SAFE-FULL-100EP`):
  - `experiments/results/camponotus_rfdetr_sequence_safe_val_vs_yolo.json`
  - `experiments/results/camponotus_rfdetr_sequence_safe_test_vs_yolo.json`

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | Precision | Recall | TP/FP/FN | FPS | Latency mean (ms) |
|-------|---------------:|--------:|--------:|----------:|-------:|----------:|----:|-------------------:|
| Val | 0.261 | 0.467 | 0.288 | 0.589 | 0.532 | 2495/1741/2193 | 11.3 | 88.5 |
| Test | 0.364 | 0.777 | 0.260 | 0.810 | 0.858 | 938/220/155 | 16.9 | 59.1 |

**RF-DETR − YOLO deltas (same sequence-safe val/test references):**

- **Val:** mAP@[.5:.95] **+0.115**, mAP@.50 **+0.168**, matched recall **+0.063**, matched precision **−0.011**; FPS **−10.0**, latency **+41.7 ms**.
- **Test:** mAP@[.5:.95] **+0.124**, mAP@.50 **+0.259**, matched recall **+0.069**, matched precision **+0.067**; FPS **−13.4**, latency **+26.1 ms**.

**Interpretation (draft):**

- On this Camponotus sequence-safe setup, RF-DETR improves detection quality over the recorded YOLO baseline on both val and test (higher mAP and recall, especially strong test gains).
- The speed trade-off is substantial: RF-DETR is notably slower in this run (lower FPS, higher latency).
- For reporting, this is a clear quality-vs-throughput trade-off: RF-DETR currently looks better for offline/high-accuracy analysis, while YOLO remains stronger for real-time constraints.
- As with prior Camponotus runs, area-bucket caveats still apply (`mAP_small=-1` in both models here; `mAP_medium` is sparse/unstable across splits), so primary comparison should focus on overall mAP and matched P/R.

---

### EXP-CAMPO-RFDETR-TRACKIDMAJORITY-FULL — RF-DETR Small (Idea 1, 2-class state)

**Goal:** Train/evaluate RF-DETR on the `track_id`-majority split (same split policy as the track_id-majority YOLO run) and compare quality/speed against both YOLO and the RF-DETR sequence-safe run.

**Artifacts:**

- Metrics:
  - `experiments/results/camponotus_rfdetr_trackidmajor_val_metrics.json`
  - `experiments/results/camponotus_rfdetr_trackidmajor_test_metrics.json`
- RF-DETR vs YOLO compare:
  - `experiments/results/camponotus_rfdetr_trackidmajor_val_vs_yolo.json`
  - `experiments/results/camponotus_rfdetr_trackidmajor_test_vs_yolo.json`

**YOLO baseline for recorded RF-DETR deltas:** the bundled compare JSONs above use **`camponotus_idea1_trackidmajor_full_40ep_b8w4`** (**640**). **RF-DETR vs YOLO @896** (same resolution intent) is recorded under **EXP-CAMPO-RFDETR-TRACKIDMAJORITY-896** — see [`camponotus_rfdetr_trackidmajor_896_vs_yolo896_{val,test}.json`](../experiments/results/camponotus_rfdetr_trackidmajor_896_vs_yolo896_val.json).

**Quantitative results (track_id-majority split):**

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | Precision | Recall | TP/FP/FN | FPS | Latency mean (ms) |
|-------|---------------:|--------:|--------:|----------:|-------:|----------:|----:|-------------------:|
| Val | 0.427 | 0.727 | 0.450 | 0.751 | 0.869 | 2385/791/358 | 12.1 | 82.6 |
| Test | 0.327 | 0.600 | 0.349 | 0.564 | 0.462 | 1679/1299/1956 | 8.9 | 112.6 |

**RF-DETR − YOLO deltas (same track_id-majority val/test references):**

- **Val:** mAP@[.5:.95] **+0.062**, mAP@.50 **+0.089**, matched recall **+0.102**, matched precision **−0.064**; FPS **−25.1**, latency **+55.7 ms**.
- **Test:** mAP@[.5:.95] **+0.034**, mAP@.50 **+0.075**, matched recall **+0.068**, matched precision **−0.100**; FPS **−22.1**, latency **+80.3 ms**.

**RF-DETR track_id-majority vs RF-DETR sequence-safe (cross-split diagnostic):**

- **Val (track_id-majority minus sequence-safe):** mAP@[.5:.95] **+0.166**, mAP@.50 **+0.260**, precision **+0.162**, recall **+0.337**, FPS **+0.8**, latency **−6.0 ms**.
- **Test (track_id-majority minus sequence-safe):** mAP@[.5:.95] **−0.037**, mAP@.50 **−0.177**, precision **−0.246**, recall **−0.396**, FPS **−8.0**, latency **+53.5 ms**.
- **Interpretation:** within RF-DETR, the track_id-majority split looks much easier on val but clearly harder on test (and slower). Treat this as split-policy sensitivity, not a model-family ranking by itself.
- **Qualitative finding (video-level behavior):** in manual review, the track_id-majority setup often keeps short temporal windows more identity-consistent, so trophallaxis episodes can appear more stable and less frequently "overwritten" by nearby ant frames. Keep this as a complementary diagnostic signal: useful for interaction-readability analysis, but not a replacement for stricter held-out sequence-safe generalization metrics.

---

### EXP-CAMPO-RFDETR-TRACKIDMAJORITY-896 — RF-DETR Small (Idea 1, train/infer @896)

**Goal:** Train RF-DETR at **`resolution=896`** on the **`track_id`-majority** split and compare to **YOLO26n @896** on the **same** GT and image roots — controlled input scale vs the prior **640** RF-DETR vs YOLO table.

**Training:** [`experiments/rfdetr/camponotus_rfdetr_trackidmajor_896/config.yaml`](../experiments/rfdetr/camponotus_rfdetr_trackidmajor_896/config.yaml) — `RFDETRSmall`, `epochs=30`, `batch_size=4`, `grad_accum_steps=4`, `resolution=896`, `seed=42`. **Trainer-native val** (epoch 29): best regular mAP **0.5130** → checkpoint `checkpoint_best_regular.pth`; weights copied to **`experiments/rfdetr/camponotus_rfdetr_trackidmajor_896/weights/best.pth`**. **Paper-facing** metrics below are **`evaluate.py`** (pycocotools + matched P/R), not the trainer table.

**Data:** Roboflow-style export under `datasets/camponotus_rfdetr_coco_trackidmajor/` from [`prepare_camponotus_coco_rfdetr.py`](../scripts/datasets/prepare_camponotus_coco_rfdetr.py) (see manifest `camponotus_rfdetr_manifest.json`).

**Inference / eval:** `infer_rfdetr.py` → `bench_rfdetr.py` → `evaluate.py` with **`--inference-benchmark-json`**; GT/images under `datasets/camponotus_coco/.../trackidmajor/` and `datasets/camponotus_yolo/camponotus_full_export_unique_trackidmajor/images/{val,test}`.

**Artifacts (unified `evaluate.py`):**

- Val: [`experiments/results/camponotus_rfdetr_trackidmajor_896_metrics_val.json`](../experiments/results/camponotus_rfdetr_trackidmajor_896_metrics_val.json)
- Test: [`experiments/results/camponotus_rfdetr_trackidmajor_896_metrics_test.json`](../experiments/results/camponotus_rfdetr_trackidmajor_896_metrics_test.json)
- Predictions / bench: `experiments/results/camponotus_rfdetr_trackidmajor_896_predictions_{val,test}.json`, `…_bench_{val,test}.json`

**896 RF-DETR quantitative results (matched P/R @ IoU≥0.5, score≥0.25):**

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | Precision | Recall | TP/FP/FN | FPS | Latency mean (ms) |
|-------|---------------:|--------:|--------:|----------:|-------:|----------:|----:|------------------:|
| Val | 0.356 | 0.650 | 0.341 | 0.745 | 0.855 | 2346/804/397 | 25.1 | 39.9 |
| Test | 0.325 | 0.575 | 0.343 | 0.548 | 0.407 | 1480/1220/2155 | 18.1 | 55.2 |

**RF-DETR − YOLO896** (same split; baseline = `camponotus_idea1_trackidmajor_full_896_metrics_*`; Δ = RF-DETR − YOLO). Bundles: [`camponotus_rfdetr_trackidmajor_896_vs_yolo896_val.json`](../experiments/results/camponotus_rfdetr_trackidmajor_896_vs_yolo896_val.json), [`…_test.json`](../experiments/results/camponotus_rfdetr_trackidmajor_896_vs_yolo896_test.json).

| Split | mAP@[.5:.95] | mAP@.50 | mAP_medium | Matched ΔP | Matched ΔR | FPS Δ | Latency mean Δ (ms) |
|-------|-------------:|--------:|-----------:|-----------:|-----------:|------:|---------------------:|
| Val | **+0.038** | **+0.013** | **−0.030** | −0.023 | **+0.050** | **−28.3** | **+21.2** |
| Test | **+0.032** | **+0.038** | 0.000* | **−0.143** | **−0.084** | **−23.4** | **+31.1** |

\*Test **mAP_medium = 0** for both runs; **0.000** is not a substantive “no change in medium objects.”

**Regenerate compare JSONs:**

```bash
python3 scripts/evaluation/compare_camponotus_rfdetr_vs_yolo.py \
  --baseline experiments/results/camponotus_idea1_trackidmajor_full_896_metrics_val.json \
  --compare experiments/results/camponotus_rfdetr_trackidmajor_896_metrics_val.json \
  --out experiments/results/camponotus_rfdetr_trackidmajor_896_vs_yolo896_val.json

python3 scripts/evaluation/compare_camponotus_rfdetr_vs_yolo.py \
  --baseline experiments/results/camponotus_idea1_trackidmajor_full_896_metrics_test.json \
  --compare experiments/results/camponotus_rfdetr_trackidmajor_896_metrics_test.json \
  --out experiments/results/camponotus_rfdetr_trackidmajor_896_vs_yolo896_test.json
```

**Interpretation (draft):** At **896**, RF-DETR **improves COCO mAP** (@[.5:.95] and @.50) vs YOLO on **both** val and test — consistent with the **640** Camponotus RF-DETR advantage, now at matched **train/infer resolution** to YOLO896. **Throughput** remains strongly in YOLO’s favor (FPS down ~23–28, latency up ~21–31 ms on RTX 4070). **Val** greedy matched **recall** rises vs YOLO896 at a small **precision** cost. **Test** shows **higher mAP** for RF-DETR but **lower** greedy matched P/R and **fewer TPs** than YOLO in this run — COCO AP and greedy IoU≥0.5 matching can disagree when score/calibration differs; treat **test** readouts as requiring both **mAP** and **matched P/R** side by side. **Within RF-DETR:** comparing **896** to the recorded **640** track_id-majority RF-DETR run is a **resolution + schedule** change (30 ep @896 vs prior RF-DETR full run), not a single-variable ablation.

### EXP-CAMPO-RFDETR-SEQUENCE-SAFE-896 — RF-DETR Small (Idea 1, train/infer @896)

**Goal:** Fill the sequence-safe **RF-DETR @896** rows in the Idea 1 matrix and compare against both YOLO baselines (**@896** and legacy **@640/100ep**) plus prior RF-DETR sequence-safe **@640**.

**Artifacts (unified `evaluate.py`):**

- Metrics: `experiments/results/camponotus_rfdetr_sequence_safe_896_metrics_{val,test}.json`
- Predictions / bench: `experiments/results/camponotus_rfdetr_sequence_safe_896_predictions_{val,test}.json`, `experiments/results/camponotus_rfdetr_sequence_safe_896_bench_{val,test}.json`
- Compare bundles:
  - vs YOLO896: `camponotus_rfdetr_sequence_safe_896_vs_yolo896_{val,test}.json`
  - vs YOLO640: `camponotus_rfdetr_sequence_safe_896_vs_yolo640_{val,test}.json`
  - vs RF-DETR640: `camponotus_rfdetr_sequence_safe_896_vs_640_{val,test}.json`

**RF-DETR @896 quantitative results (matched P/R @ IoU≥0.5, score≥0.25):**

| Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | Precision | Recall | TP/FP/FN | FPS | Latency mean (ms) |
|-------|---------------:|--------:|--------:|----------:|-------:|----------:|----:|------------------:|
| Val | 0.208 | 0.384 | 0.197 | 0.535 | 0.472 | 2215/1925/2473 | 11.5 | 86.8 |
| Test | 0.311 | 0.712 | 0.235 | 0.760 | 0.848 | 927/292/166 | 16.9 | 59.3 |

**RF-DETR896 − YOLO896 (sequence-safe):**

| Split | mAP@[.5:.95] | mAP@.50 | mAP_medium | Matched ΔP | Matched ΔR | FPS Δ | Latency mean Δ (ms) |
|-------|-------------:|--------:|-----------:|-----------:|-----------:|------:|---------------------:|
| Val | **+0.020** | **+0.004** | 0.000* | −0.088 | −0.066 | **−21.7** | **+56.7** |
| Test | **+0.144** | **+0.360** | **+0.300** | **+0.115** | **+0.143** | **−24.7** | **+35.2** |

\*Val `mAP_medium=0` for both runs; delta 0.000 is not a substantive medium-bucket finding.

**RF-DETR896 − YOLO640 (sequence-safe legacy baseline):** Val mAP@[.5:.95] **+0.062**, mAP@.50 **+0.085**, FPS **−9.8**, latency **+39.9 ms**. Test mAP@[.5:.95] **+0.072**, mAP@.50 **+0.194**, FPS **−13.4**, latency **+26.2 ms**.

**RF-DETR896 − RF-DETR640 (same split, compare − baseline):** Val mAP@[.5:.95] **−0.053**, mAP@.50 **−0.083**, matched P/R **−0.054 / −0.060**, FPS **+0.23**. Test mAP@[.5:.95] **−0.053**, mAP@.50 **−0.065**, matched P/R **−0.050 / −0.010**, FPS **−0.04** (essentially flat).

**Interpretation (draft):** On the sequence-safe split, RF-DETR896 strongly beats YOLO896 on **test** quality (both AP and matched P/R), but not on throughput (much slower). Compared with prior RF-DETR640 sequence-safe, the new 896 run is **lower** on val/test AP and slightly lower on matched P/R, with near-identical speed; treat it as a schedule/config-sensitive result rather than a pure “higher resolution helps” claim.

### EXP-CAMPO-PRELABEL-TRACKING-001 — ByteTrack prelabel ablation for Idea 1 dataset workflow

**Goal:** Improve CVAT prelabel usability for Idea 1 (two-class detection) by adding stable `track_id` while avoiding excessive box loss.

**Inputs and outputs (recorded run):**

- ordinary prelabels: `datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_plain.json`
- tracked baseline: `datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked.json`
- tracked tunedA: `datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_tunedA.json`
- compare JSONs:
  - `experiments/results/camponotus_prelabels_tracking_compare_baseline.json`
  - `experiments/results/camponotus_prelabels_tracking_compare_tunedA.json`

**Tracking settings compared:**

- **Baseline:** `track_thresh=0.25`, `match_thresh=0.8`, `track_buffer=30`, `min_track_len=2`
- **TunedA:** `track_thresh=0.20`, `match_thresh=0.75`, `track_buffer=45`, `min_track_len=1`

**Quantitative comparison (vs ordinary prelabels):**

| Variant | Annotations | Delta vs ordinary | Images with annotations | Track coverage | Unique tracks | Track len mean / median | Short tracks <=2 | Gap events | Gap frames |
|---------|------------:|------------------:|------------------------:|---------------:|--------------:|------------------------:|-----------------:|-----------:|----------:|
| Ordinary (no tracking) | 12791 | 0 | 1505 | 0.0% | 0 | - | - | - | - |
| Tracked baseline | 11051 | -1740 (-13.6%) | 1463 | 100.0% | 623 | 17.74 / 10.0 | 10.75% | 1465 | 5466 |
| Tracked tunedA | 10920 | -1871 (-14.6%) | 1462 | 100.0% | 785 | 13.91 / 6.0 | 23.44% | 1427 | 6444 |

**Interpretation:** Both tracking variants successfully assign persistent IDs (`track_id` coverage 100%). For Idea 1 dataset preparation, **baseline** is preferable to tunedA: tunedA slightly reduces gap-event count but increases total missing-gap frames, produces substantially more short fragmented tracks, and drops more boxes vs ordinary prelabels.

**Working recommendation (Idea 1):** Keep ByteTrack enabled for CVAT bootstrap, but use the **baseline** settings above as default. Continue manual correction in CVAT as the quality gate.

**ID stability finding (current):** For dense Camponotus scenes, even with a good detector baseline (YOLO26n_v2) and standard tracking libraries (ByteTrack in this run), it is **not realistic to fully eliminate true-positive ID flicker** (`track_id` switches/fragmentation). Tracking reduces manual work and improves continuity, but some TP ID instability remains and must be handled by annotation QA. This is expected behavior in heavy occlusion/contact scenarios and should be treated as an operational constraint, not as a pipeline failure.

### EXP-CAMPO-PRELABEL-TRACKING-002 — BoT-SORT (+ReID) follow-up for Idea 1 dataset workflow

**Goal:** Test whether BoT-SORT with appearance ReID improves ID continuity vs ByteTrack while preserving usable prelabels for CVAT.

**Inputs and outputs (recorded run):**

- ordinary prelabels: `datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_plain.json`
- BoT-SORT(+ReID) prelabels: `datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_botsort_reid.json`
- compare JSON: `experiments/results/camponotus_prelabels_tracking_compare_botsort_reid.json`

**Tracking settings (BoT-SORT run):** `track_thresh=0.25`, `match_thresh=0.8`, `track_buffer=30`, `min_track_len=2`, `tracker=botsort`, `with_reid=true`.

**Quantitative comparison (vs ordinary prelabels):**

| Variant | Annotations | Delta vs ordinary | Images with annotations | Track coverage | Unique tracks | Track len mean / median | Short tracks <=2 | Gap events | Gap frames |
|---------|------------:|------------------:|------------------------:|---------------:|--------------:|------------------------:|-----------------:|-----------:|----------:|
| Ordinary (no tracking) | 12791 | 0 | 1505 | 0.0% | 0 | - | - | - | - |
| BoT-SORT + ReID | 13517 | +726 (+5.7%) | - | 100.0% | 330 | 40.96 / 37.5 | 4.24% | 907 | 3713 |

**Interpretation:** Compared with the prior ByteTrack baseline (11051 boxes, short<=2 10.75%, gap events 1465), BoT-SORT(+ReID) substantially improves continuity indicators: fewer fragmented short tracks, longer trajectories, and fewer gap events/frames. It also retains more detections than ordinary prelabels in this run.

**Working recommendation (Idea 1):** Prefer **BoT-SORT + ReID** as the default tracking option for CVAT bootstrap on this dataset, then validate prelabel precision on a small hard-clip sample during annotation QA.

### EXP-CAMPO-PRELABEL-TRACKING-003 — Soft state-priority relabel toggle (tooling update)

**Goal:** Reduce confusing class conflicts in dense contact scenes without deleting detections.

**Implemented toggles (both prelabels + video QA):**

- `--state-priority-soft`
- `--state-priority-iou-thresh` (default `0.70`)
- `--state-priority-score-gap-max` (default `0.12`)

**Rule:** if a `normal` box strongly overlaps a `trophallaxis` box and confidence scores are close, relabel `normal -> trophallaxis`. The box is kept (no suppression/drop).

**Why this is safer than hard suppression:** the previous drop-based overlap filter removed potentially valid ant detections and did not improve practical annotation quality. The soft rule preserves geometry and only changes class in ambiguous overlap cases.

**Current status:** tooling added; quantitative effect should be measured on the next tracked-vs-soft compare run and then recorded here.

### EXP-CAMPO-PRELABEL-TRACKING-004 — CVAT Video 1.1 XML export + timeline/interpolation fix (tooling update)

**Goal:** Provide a tracking-native CVAT import path that preserves per-box `state` while avoiding overdraw/track explosion artifacts during review.

**Implemented outputs/toggles:**

- `bootstrap_camponotus_autolabel.py --cvat-video-xml-out ...` (CVAT Video 1.1 XML sidecar)
- exporter supports `<track>` + per-frame `<box>` and `attribute name="state"` (`normal` / `trophallaxis`)

**Root cause observed during first XML iteration:** CVAT showed many overlapping/long-lived tracks caused by timeline/interpolation mismatch (frame indexing + missing explicit track closure on gaps).

**Fix applied:** XML export now uses a consistent global frame timeline and writes explicit `outside=1` closure keyframes when a track disappears (gap/end), preventing unintended interpolation carry-over.

**Related analytics consistency fix:** `track_yolo_video.py` frame indexing was corrected so `frame_index` increments only for valid rendered frames (prevents gap/length metric skew if `orig_img` is `None`).

**Current status:** validated qualitatively in CVAT; keep this XML path as the default when native track editing is required. COCO+`track_id` remains useful for training/evaluation metadata.

---

### Idea 1 — Paper-ready summary (YOLO26n vs RF-DETR Small)

**Headline readout (generalization):** For claims about **sequence-held-out** generalization, prioritize **sequence-safe test** metrics (same GT/images under `…/camponotus_full_export_unique_sequence_safe/`). Treat **track_id–majority** as a **supplementary** split (different val/test images); use it for within-split comparisons, not as a drop-in replacement for sequence-safe.

**Convention:** All numbers from unified [`evaluate.py`](../scripts/evaluation/evaluate.py) (COCO mAP + matched P/R @ IoU≥0.5, score≥0.25; RTX 4070 where recorded). Rows are **not** controlled for training budget — see footnotes and per-experiment subsections above.

#### Sequence-safe split

| Model | Split | mAP@[.5:.95] | mAP@.50 | P | R | FPS | Latency mean (ms) |
|-------|-------|-------------:|--------:|--:|--:|----:|------------------:|
| YOLO26n @640 | Val | 0.146 | 0.299 | 0.600 | 0.469 | 21.3 | 46.9 |
| YOLO26n @896 | Val | 0.188 | 0.380 | 0.623 | 0.538 | 33.2 | 30.1 |
| RF-DETR Small @640 | Val | 0.261 | 0.467 | 0.589 | 0.532 | 11.3 | 88.5 |
| RF-DETR Small @896 | Val | 0.208 | 0.384 | 0.535 | 0.472 | 11.5 | 86.8 |
| YOLO26n @640 | Test | 0.240 | 0.518 | 0.743 | 0.790 | 30.3 | 33.0 |
| YOLO26n @896 | Test | 0.167 | 0.353 | 0.645 | 0.705 | 41.5 | 24.1 |
| RF-DETR Small @640 | Test | 0.364 | 0.777 | 0.810 | 0.858 | 16.9 | 59.1 |
| RF-DETR Small @896 | Test | 0.311 | 0.712 | 0.760 | 0.848 | 16.9 | 59.3 |

**Metrics JSONs (sequence-safe):** YOLO640 `camponotus_idea1_sequence_safe_full_100ep_metrics_{val,test}.json`; YOLO896 `camponotus_idea1_sequence_safe_full_896_metrics_{val,test}.json`; RF-DETR640 `camponotus_rfdetr_sequence_safe_{val,test}_metrics.json`; RF-DETR896 `camponotus_rfdetr_sequence_safe_896_metrics_{val,test}.json`. **RF-DETR896 compare bundles:** `camponotus_rfdetr_sequence_safe_896_vs_yolo896_{val,test}.json`, `camponotus_rfdetr_sequence_safe_896_vs_yolo640_{val,test}.json`, `camponotus_rfdetr_sequence_safe_896_vs_640_{val,test}.json`.

#### track_id–majority split

| Model | Split | mAP@[.5:.95] | mAP@.50 | P | R | FPS | Latency mean (ms) |
|-------|-------|-------------:|--------:|--:|--:|----:|------------------:|
| YOLO26n @640 | Val | 0.365 | 0.638 | 0.815 | 0.767 | 37.2 | 26.8 |
| YOLO26n @896 | Val | 0.319 | 0.637 | 0.768 | 0.805 | 53.4 | 18.7 |
| RF-DETR Small @640 | Val | 0.427 | 0.727 | 0.751 | 0.869 | 12.1 | 82.6 |
| RF-DETR Small @896 | Val | 0.356 | 0.650 | 0.745 | 0.855 | 25.1 | 39.9 |
| YOLO26n @640 | Test | 0.293 | 0.525 | 0.664 | 0.394 | 31.0 | 32.3 |
| YOLO26n @896 | Test | 0.293 | 0.536 | 0.691 | 0.491 | 41.5 | 24.1 |
| RF-DETR Small @640 | Test | 0.327 | 0.600 | 0.564 | 0.462 | 8.9 | 112.6 |
| RF-DETR Small @896 | Test | 0.325 | 0.575 | 0.548 | 0.407 | 18.1 | 55.2 |

**Metrics JSONs (track_id–majority):** YOLO640 `camponotus_idea1_trackidmajor_full_40ep_b8w4_metrics_{val,test}.json`; YOLO896 `camponotus_idea1_trackidmajor_full_896_metrics_{val,test}.json`; RF-DETR640 `camponotus_rfdetr_trackidmajor_{val,test}_metrics.json`; RF-DETR896 `camponotus_rfdetr_trackidmajor_896_metrics_{val,test}.json`. **Vs YOLO @896:** `camponotus_rfdetr_trackidmajor_896_vs_yolo896_{val,test}.json`.

**Footnotes:** (1) **YOLO sequence-safe @640** = 100 epochs, `imgsz=640`, batch 4, workers 0; **YOLO sequence-safe @896** = 40 epochs, `imgsz=896`, batch 8, workers 4, early stop — not comparable as a pure resolution sweep. (2) **YOLO track_id–majority @640 vs @896** uses matched early-stop schedule for 896 (`b8w4`); see **EXP-CAMPO-IDEA1-TRACKIDMAJORITY-FULL-896**. (3) **RF-DETR @896 (track_id–majority)** = 30 epochs, `batch_size=4`, `grad_accum_steps=4`, `resolution=896` (**EXP-CAMPO-RFDETR-TRACKIDMAJORITY-896**). (4) Cross-split **sequence-safe vs track_id–majority** at the same resolution is **diagnostic only** (different images); see the cross-split subsection under YOLO896.

**Optional throughput (RF-DETR):** `infer_rfdetr.py` can enable `model.optimize_for_inference()` when the environment variable `EXP_A005_OPTIMIZE_INFERENCE` is set (same pattern as ants EXP-A005). Re-benchmark and re-evaluate if you report optimized FPS; do not assume mAP is unchanged without checking.

**Planned benchmark steps:** [`camponotus_research_roadmap.md`](camponotus_research_roadmap.md) → section **“Idea 1 — Next steps (benchmark)”**.

---

## Idea 2 — Event-level baseline template (hybrid)

Use this template for each Idea 2 run:

- **Prediction artifact:** `experiments/results/<run>_events.json`
- **Evaluation artifact:** `experiments/results/<run>_events_eval.json`
- **Compare artifact (optional):** `experiments/results/<run>_vs_<baseline>.json`

Recommended table (fill after each run):

| Run | Helper signal | tIoU match thr | Precision | Recall | F1 | Mean tIoU (matched) | TP | FP | FN |
|-----|---------------|----------------|----------:|-------:|---:|---------------------:|---:|---:|---:|
| `<run_name>` | on/off | 0.30 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Per-clip reporting:** include the worst clips by F1 and top identity-pair misses from the evaluator `per_sequence` block.

**Protocol reference:** [`camponotus_idea2_event_protocol.md`](camponotus_idea2_event_protocol.md).

---

## Changelog

| Date | Experiment(s) | Summary |
|------|----------------|---------|
| 2026-03-31 | Idea 2 hybrid baseline tooling (phases 1-5 scaffold) | Added event protocol doc (`camponotus_idea2_event_protocol.md`), frozen benchmark subset scaffold (`datasets/camponotus_idea2_event_benchmark_v1.json`), and runnable scripts for event inference/evaluation/compare: `infer_camponotus_idea2_events.py`, `evaluate_camponotus_idea2_events.py`, `compare_camponotus_idea2_event_metrics.py`. Added reproducible Idea 2 CLI recipes and event-results template sections for ongoing experiments. |
| 2026-03-31 | EXP-CAMPO-RFDETR-SEQUENCE-SAFE-896 (RF-DETR Small) | Sequence-safe RF-DETR @896 (`camponotus_rfdetr_sequence_safe_896_metrics_{val,test}.json`): **val** mAP@[.5:.95]=0.208, mAP@.50=0.384, matched P/R=0.535/0.472 (TP/FP/FN 2215/1925/2473), FPS 11.5, latency 86.8 ms; **test** mAP@[.5:.95]=0.311, mAP@.50=0.712, matched P/R=0.760/0.848 (927/292/166), FPS 16.9, latency 59.3 ms. **Vs YOLO896:** val ΔmAP@[.5:.95] +0.020 (P/R down), test ΔmAP@[.5:.95] +0.144 with P/R up; throughput slower by ~21.7 FPS (val) and ~24.7 FPS (test). **Vs RF-DETR640:** mAP drops ~0.053 on both val/test with near-flat speed. |
| 2026-03-31 | OOD qualitative diagnostic (Idea 1) | New-lab clip `trophalaxis_001_example.mp4` (455 frames, expected mostly trophallaxis) with YOLO track_id-majority @896 + soft state-priority (`--conf 0.25`) still outputs **normal-dominant** counts: `normal=698`, `trophallaxis=262`, `unique_tracks=31`, `relabel_count=9`. New `per_track` analytics reveals asymmetric long identities: `id 1` = 211 troph / 159 normal vs `id 2` = 442 normal / 1 troph. Interpretation: likely label-policy/domain-shift limitation for Idea 1 two-class frame labels; add label-audit checklist and treat this as failure analysis while planning Idea 2 event-level modeling. Artifact: `experiments/visualizations/camponotus_idea1_trackidmajor_full_896_tracked_camponotus_trophalaxis_001_soft_analytics.json`. |
| 2026-03-31 | Documentation | **Idea 1 paper-ready summary:** two matrices (sequence-safe + track_id–majority) for YOLO640 / YOLO896 / RF-DETR640 / RF-DETR896 with mAP, matched P/R, FPS, latency — **including RF-DETR @896 sequence-safe** (`camponotus_rfdetr_sequence_safe_896_metrics_{val,test}.json`, see **EXP-CAMPO-RFDETR-SEQUENCE-SAFE-896**); headline readout = sequence-safe **test**; footnotes on schedule mismatch; pointer to optional `EXP_A005_OPTIMIZE_INFERENCE` for RF-DETR latency; **Next steps** in `camponotus_research_roadmap.md`. |
| 2026-03-31 | EXP-CAMPO-RFDETR-TRACKIDMAJORITY-896 (RF-DETR Small) | Track_id-majority @ **896** (`epochs=30`, `batch=4`, `grad_accum=4`, best epoch **29**): **val** mAP@[.5:.95]=0.356, mAP@.50=0.650, matched P/R=0.745/0.855 (2346/804/397), FPS 25.1, latency 39.9 ms. **Test:** mAP@[.5:.95]=0.325, mAP@.50=0.575, matched P/R=0.548/0.407 (1480/1220/2155), FPS 18.1, latency 55.2 ms. **Vs YOLO896:** val ΔmAP@[.5:.95] **+0.038**, matched R **+0.050**, FPS **−28.3**; test ΔmAP@[.5:.95] **+0.032**, matched P/R **−0.143 / −0.084**, FPS **−23.4**. Metrics: `camponotus_rfdetr_trackidmajor_896_metrics_{val,test}.json`. Compare: `camponotus_rfdetr_trackidmajor_896_vs_yolo896_{val,test}.json`. |
| 2026-03-30 | Documentation | Added Camponotus **prior work** block: Greenwald *et al.* (2015) *Sci. Rep.* trophallaxis / fluorescence + barcode citation, discussion contrast vs this repo’s video-only benchmark; BibTeX. Cross-link from `camponotus_research_roadmap.md`. |
| 2026-03-30 | Cross-split diagnostic (YOLO896) | `compare_metrics.py`: track_id–majority **896** minus sequence-safe **896** on val/test **labels** only (different images). **Val Δ:** mAP@[.5:.95] **+0.130**, mAP@.50 **+0.257**, matched P/R **+0.144 / +0.267**, FPS **+20.2**. **Test Δ:** mAP@[.5:.95] **+0.126**, mAP@.50 **+0.184**, matched P **+0.046**, R **−0.214**, FPS ~flat. Files: `camponotus_crosssplit_seqsafe896_vs_trackid896_{val,test}.json`. |
| 2026-03-30 | EXP-CAMPO-IDEA1-TRACKIDMAJORITY-FULL-896 (YOLO26n) | Track_id-majority Idea 1 @ **896** (same schedule as 640 b8w4: early stop at 35 epochs, best epoch 20): **val** mAP@[.5:.95]=0.319, mAP@.50=0.637, matched P/R=0.768/0.805 (2208/668/535), FPS 53.4, latency 18.7 ms. **Test:** mAP@[.5:.95]=0.293, mAP@.50=0.536, matched P/R=0.691/0.491 (1785/799/1850), FPS 41.5, latency 24.1 ms. **Vs 640 b8w4:** val ΔmAP@[.5:.95] **−0.047** (mAP@.50 ~flat), matched R **+0.038** / P **−0.048**, FPS **+16.2**; test ΔmAP@[.5:.95] **~0**, mAP@.50 **+0.012**, matched R **+0.097**, FPS **+10.5**. Metrics: `camponotus_idea1_trackidmajor_full_896_metrics_{val,test}.json`. Compare: `camponotus_trackidmajor_896_vs_640_{val,test}.json`. |
| 2026-03-30 | EXP-CAMPO-IDEA1-SEQUENCE-SAFE-FULL-896 (YOLO26n) | Sequence-safe Idea 1 @ **896** (`epochs=40`, `batch=8`, `workers=4`, early stop): **val** mAP@[.5:.95]=0.188, mAP@.50=0.380, matched P/R=0.623/0.538 (2523/1525/2165), FPS 33.2, latency 30.1 ms. **Test:** mAP@[.5:.95]=0.167, mAP@.50=0.353, matched P/R=0.645/0.705 (771/424/322), FPS 41.5, latency 24.1 ms. **Vs 640/100ep:** val ΔmAP@[.5:.95] **+0.042**, test **−0.073**; val/test throughput faster vs recorded 640 bench. Metrics: `camponotus_idea1_sequence_safe_full_896_metrics_{val,test}.json`. **`compare_metrics.py` bundles:** `camponotus_sequence_safe_896_vs_640_100ep_{val,test}.json`. |
| 2026-03-30 | EXP-CAMPO-RFDETR-TRACKIDMAJORITY-FULL (RF-DETR Small) | RF-DETR on track_id-majority split. **Val:** mAP@[.5:.95]=0.427, mAP@.50=0.727, matched P/R=0.751/0.869 (TP/FP/FN 2385/791/358), FPS 12.1, latency 82.6 ms. **Test:** mAP@[.5:.95]=0.327, mAP@.50=0.600, matched P/R=0.564/0.462 (1679/1299/1956), FPS 8.9, latency 112.6 ms. Vs YOLO track_id-majority baseline: val ΔmAP@[.5:.95] +0.062, test +0.034; throughput much slower (val FPS −25.1, test FPS −22.1). |
| 2026-03-30 | EXP-CAMPO-RFDETR-SEQUENCE-SAFE-FULL (RF-DETR Small) | First recorded Camponotus RF-DETR sequence-safe run. **Val:** mAP@[.5:.95]=0.261, mAP@.50=0.467, matched P/R=0.589/0.532 (TP/FP/FN 2495/1741/2193), FPS 11.3, latency 88.5 ms. **Test:** mAP@[.5:.95]=0.364, mAP@.50=0.777, matched P/R=0.810/0.858 (938/220/155), FPS 16.9, latency 59.1 ms. Vs YOLO sequence-safe baseline: val ΔmAP@[.5:.95] +0.115, test +0.124; throughput slower (val FPS −10.0, test FPS −13.4). |
| 2026-03-30 | EXP-CAMPO-IDEA1-TRACKIDMAJORITY-FULL-40EP-B8W4 (YOLO26n) | Track_id-majority full run with `epochs=40`, `batch=8`, `workers=4`, `patience=15` (early stop at epoch 35, best epoch 20): val mAP@[.5:.95]=0.365 (mAP@.50=0.638), test mAP@[.5:.95]=0.293 (mAP@.50=0.525); matched P/R: val 0.815/0.767 (TP/FP/FN 2105/477/638), test 0.664/0.394 (1431/724/2204); FPS/latency mean: val 37.2 / 26.8 ms, test 31.0 / 32.3 ms. Strong gain vs prior 1-epoch track_id-majority smoke (+0.243 val and +0.280 test mAP@[.5:.95]). |
| 2026-03-30 | EXP-CAMPO-IDEA1-TRACKIDMAJORITY-SMOKE-001 (YOLO26n) | Smoke 1 epoch at `imgsz=640` on `track_id`-majority split with unique basenames: val mAP@[.5:.95]=0.122 (mAP@.50=0.235), test mAP@[.5:.95]=0.013 (mAP@.50=0.045); matched P/R: val 0.798/0.439, test 0.590/0.109; FPS/latency mean: val 35.9 FPS / 27.8 ms, test 29.8 FPS / 33.5 ms; split leakage QA: 40/638 overlapping `track_id`s. |
| 2026-03-30 | EXP-CAMPO-IDEA2-ANTONLY-TRACKIDMAJORITY-SMOKE-002 (YOLO26n) | Ant-only derivative from Idea 1 with `trophallaxis_gt` side-signal: smoke 1 epoch at `imgsz=640` on same split; val mAP@[.5:.95]=0.220 (mAP@.50=0.478), test mAP@[.5:.95]=0.059 (mAP@.50=0.139); matched P/R: val 0.753/0.541, test 0.642/0.166; FPS/latency mean: val 35.4 FPS / 28.2 ms, test 40.8 FPS / 24.5 ms; detection-only smoke (interaction side-signal not used in `evaluate.py`). |
| 2026-03-30 | EXP-CAMPO-IDEA1-SEQUENCE-SAFE-FULL-100EP (YOLO26n) | Full 100 epochs at `imgsz=640` on sequence-safe split: val mAP@[.5:.95]=0.146 (mAP@.50=0.299), test mAP@[.5:.95]=0.240 (mAP@.50=0.518); matched P/R @IoU0.5: val 0.600/0.469 (TP/FP/FN 2199/1465/2489), test 0.743/0.790 (863/298/230); FPS/latency mean: val 21.3 FPS / 46.9 ms, test 30.3 FPS / 33.0 ms; training val mAP50-95 peaked around epoch ~19 then flattened. |
| 2026-03-26 | EXP-CAMPO-PRELABEL-TRACKING-004 (tooling) | Added CVAT Video 1.1 XML export (`--cvat-video-xml-out`) with `state` attribute support and fixed timeline/interpolation behavior (global frame indexing + explicit `outside=1` closures) to avoid track overdraw/explosion in CVAT. Also fixed `track_yolo_video.py` analytics frame indexing for `None` frames. |
| 2026-03-26 | EXP-CAMPO-PRELABEL-TRACKING-003 (tooling) | Added optional soft state-priority relabel to prelabel and video tracking scripts: `--state-priority-soft`, `--state-priority-iou-thresh`, `--state-priority-score-gap-max`. Rule relabels overlapping ambiguous `normal` boxes to `trophallaxis` without deleting detections; intended as a safer alternative to prior hard suppression. Quantitative results pending next run. |
| 2026-03-26 | EXP-CAMPO-PRELABEL-TRACKING-002 | BoT-SORT(+ReID) follow-up on Camponotus prelabels (`tracker=botsort`, `with_reid=true`, `0.25/0.8/30/min_len=2`): vs ordinary, annotations `12791→13517` (+5.7%), track coverage 100%, unique tracks 330, track len mean/median `40.96/37.5`, short tracks <=2 `4.24%`, gap events `907` (gap frames 3713). Compared to prior ByteTrack baseline, continuity/fragmentation metrics improved strongly; recommendation updated to prefer BoT-SORT+ReID for Idea 1 CVAT bootstrap, with QA on hard clips. |
| 2026-03-26 | EXP-CAMPO-PRELABEL-TRACKING-001 (interpretation update) | Added explicit ID-stability finding: in dense in-situ ant scenes, TP `track_id` flicker cannot be fully removed with detector+tracker alone (current evidence from ByteTrack baseline/tunedA); manual QA remains required. This baseline interpretation is now complemented by the BoT-SORT(+ReID) follow-up row above. |
| 2026-03-26 | EXP-CAMPO-PRELABEL-TRACKING-001 | Compared ordinary vs ByteTrack prelabels for Camponotus Idea 1 dataset workflow. Baseline tracking (`0.25/0.8/30/min_len=2`) produced `11051` boxes (−13.6% vs ordinary `12791`) with 100% `track_id` coverage, 623 tracks, median track length 10, short tracks <=2 at 10.75%, gap events 1465 (5466 frames). TunedA (`0.20/0.75/45/min_len=1`) underperformed for curation quality: fewer boxes (`10920`), more fragmentation (short <=2 at 23.44%), lower median track length (6), and higher total gap frames (6444). Recommendation: keep baseline tracking defaults for prelabel bootstrap. |
| 2026-03-26 | EXP-CAMPO-001-V2 (Camponotus YOLO26n retrain) | Updated dataset to **280 images** (196/42/42) with two additional in-situ videos; retrained `camponotus_yolo26n_v2`. Results: **val** mAP@[.5:.95] **0.843**, mAP@.50 **0.962**, matched P/R **0.908 / 0.961**; **test** mAP@[.5:.95] **0.902**, mAP@.50 **0.983**, matched P/R **0.920 / 0.981**; FPS ~**47.3–47.7** @640, latency ~**21 ms**. Note: inference logs reported extra stale files in val/test image dirs not present in COCO GT; skipped by `infer_yolo.py`. |
| 2026-03-25 | EXP-CAMPO-RFDETR (tooling) | Added `prepare_camponotus_coco_rfdetr.py`, `configs/expCAMPO_rfdetr.yaml`, `run_camponotus_rfdetr_exp.sh`, `infer_rfdetr.py --class-id-mode multiclass`, `compare_camponotus_rfdetr_vs_yolo.py`, `export_camponotus_ant_only_for_idea2.py`, `docs/camponotus_research_roadmap.md`. **Metrics:** run the orchestrator and fill the EXP-CAMPO-RFDETR table above. |
| 2026-03-25 | EXP-CAMPO-001 (Camponotus YOLO26n) | CVAT→prepare (`--split-source auto`)→train `camponotus_yolo26n`→infer/eval: **val** mAP@[.5:.95] **0.885**, mAP@.50 **0.935**, matched P/R **0.895 / 0.960**; **test** mAP@[.5:.95] **0.902**, mAP@.50 **0.949**, P/R **0.908 / 0.956**; FPS ~**33–34** @640 on RTX 4070. COCO small/medium AP **−1** (all boxes “large”). Caveats: auto split, small n, possible frame leakage. |
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
| 2026-03-22 | EXP-A003 (documented) | Ants SAHI pipeline: `run_ants_expA003.sh`, `expA003_ants_sahi.yaml`, `ants_expA003_vs_768.json`, `write_ants_expA003_summary.py`, `make reproduce-ants-expA003`; compare vs `ants_expA002b_imgsz768_metrics.json`. |
| 2026-03-22 | EXP-A003 (numbers) | SAHI vs vanilla 768: mAP@[.5:.95] **−0.044**, mAP_medium **−0.043**, matched P **−0.078**, R **−0.017**; TP 24183→23749, FP 2277→4651, FN 1367→1801; FPS ~60.6→~41.7, latency ~16.5→~24 ms — **prefer vanilla 768** for this setup. |
| 2026-03-22 | EXP-A003 SAHI ablation (documented) | `run_ants_expA003_sahi_ablation.py`, `evaluate.py --skip-inference-benchmark`, `ants_expA003_sahi_ablation.json` + `_summary.md`, `make reproduce-ants-expA003-ablation`. |
| 2026-03-22 | EXP-A003 SAHI ablation (numbers) | Full grid 54: **no** SAHI config beats vanilla mAP@[.5:.95] or mAP_medium; best grid ~**0.614 / 0.616** (768 slice, ov 0.15, conf 0.35, std_pred true) vs vanilla **0.645**; **lowest FP 1897** (−380 vs 2277) at 768 / ov 0.25 / conf 0.45 — precision lever only, not mAP. |
| 2026-03-21 | EXP-A004 (documented) | ANTS v1: `run_ants_expA004.sh`, `infer_ants_v1.py`, `ants_v1/` package, `bench_ants_v1.py`, `evaluate.py --inference-benchmark-json`, `compare_ants_expA004.py`, `write_ants_expA004_summary.py`, viz comparisons + `viz_ants_rois.py`, `make reproduce-ants-expA004`. |
| 2026-03-21 | EXP-A004 (numbers, RTX 4070, pre-fix compare JSON) | vs 768: mAP@[.5:.95] **0.645→0.535** (Δ **−0.110**); superseded by **fixed** bundle below (~**0.536** mAP, Δ **−0.109**). |
| 2026-03-21 | EXP-A004 (parity / merge fix) | Empty-refined passthrough; no extra NMS on stage-1-only; `predict(conf)`; `enable_dense_rois` / `pipeline_mode` / debug scripts; `make reproduce-ants-expA004-fixed`; `run_ants_expA004_staged_eval.sh`. |
| 2026-03-21 | EXP-A004 (post-fix checks + fixed metrics) | Round-trip **1073**/0; parity **50**/0; `ants_expA004_fixed_*`, `git_rev` **8e3356a**. vs 768: mAP@[.5:.95] **0.645→0.536** (Δ **−0.109**), mAP_medium **−0.106**, matched P **−0.141**, R **−0.153**; TP 24183→20286, FP 2277→5957, FN 1367→5264; FPS ~60.6→~28.9, latency ~16.5→~34.6 ms. vs SAHI: mAP **−0.064**, R **−0.136**; FPS ~41.7→~28.9 — **no** accuracy rescue vs 768/SAHI. |
| 2026-03-23 | EXP-A005 (numbers) | RF-DETR ants (unoptimized inference): mAP@[.5:.95] `0.645→0.663` (Δ `+0.018`), mAP@0.5 `0.922→0.931` (Δ `+0.009`), mAP_medium `0.645→0.664` (Δ `+0.018`); precision `0.914→0.923` (Δ `+0.009`), recall `0.946→0.962` (Δ `+0.015`); TP `24183→24568` (+385), FP `2277→2057` (-220), FN `1367→982` (-385); FPS `~60.6→~29.6` (Δ `-31.0`), latency mean `~16.5→~33.8 ms` (Δ `+17.3`). |
| 2026-03-23 | EXP-A005 (opt-infer) | RF-DETR ants (optimized inference): mAP@[.5:.95] `0.645→0.663` (Δ `+0.0183`), mAP@0.5 `0.922→0.931` (Δ `+0.0087`), mAP_medium `0.645→0.664` (Δ `+0.0185`); precision `0.914→0.923` (Δ `+0.0088`), recall `0.946→0.962` (Δ `+0.0150`); TP `24183→24567` (+384), FP `2277→2058` (-219), FN `1367→983` (-384); FPS `~60.6→~33.3` (Δ `-27.2`), latency mean `~16.5→~30.0 ms` (Δ `+13.5`). Relative to unoptimized RF-DETR: FPS `~29.6→~33.3` (+3.8), latency mean `~33.8→~30.0 ms` (-3.8). |
| 2026-03-23 | EXP-A006 (numbers) | RF-DETR + ByteTrack + smoothing vs A005 opt baseline: mAP@[.5:.95] `0.6634→0.6635` (Δ `+0.00014`), mAP@0.5 `+0.00478`, mAP_medium `+0.00053`; precision `+0.0161`, recall `-0.0063`; TP `-160`, FP `-468`, FN `+160`; FPS `33.34→30.51` (Δ `-2.83`), latency `29.99→32.77 ms` (Δ `+2.78`). Tracks: `401` total, `17` short tracks removed, avg kept length `~67.4` frames; segmentation filter disabled (`bbox` predictions). |

*(Append new rows when you re-run and refresh JSONs.)*
