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

---

## Changelog

| Date | Experiment(s) | Summary |
|------|----------------|---------|
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
