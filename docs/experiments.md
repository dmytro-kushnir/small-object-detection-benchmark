# Experiments Plan

## 🎯 Objective

Evaluate different techniques to improve small object detection.

---

## 🧪 Baseline (EXP-000)

* Model: YOLOv26
* Dataset: prepared via prepare_dataset.py
* No filtering
* Default resize
* 1 epoch (smoke test)

### Run EXP-000 (one command)

From the repo root:

```bash
chmod +x scripts/run_smoke_test.sh   # once
./scripts/run_smoke_test.sh
```

This downloads COCO128 to `datasets/raw/test/coco128`, prepares with [`configs/exp000_prepare.yaml`](../configs/exp000_prepare.yaml) into `datasets/processed/test_run`, trains `yolo26n` for one epoch to `experiments/yolo/test_run`, runs val inference, and writes **pycocotools** metrics (plus FPS/latency) to **`experiments/results/test_run_metrics.json`**.

Override device: `SMOKE_DEVICE=cpu` or `SMOKE_DEVICE=0`.

### Visualizations (after EXP-000)

From existing `predictions_val.json`, val GT, and images (no retrain):

```bash
chmod +x scripts/run_visualization.sh   # once
./scripts/run_visualization.sh
```

Writes under `experiments/visualizations/test_run/`: `predictions/` (GT green, preds red), `comparisons/` (FN / FP emphasis), `dataset/` (random GT samples; train split if prepared, else val). Default preset is EXP-000 (`test_run`). For EXP-001 use `VIZ_PRESET=exp001` or `./scripts/run_visualization.sh exp001`; for EXP-002 use `exp002` (predictions from `test_run_exp002`, val GT/images still `datasets/processed/test_run`). Override paths with `PRED`, `GT`, `IMAGES_DIR`, `VIZ_OUT`, `DATASET_GT`, `DATASET_IMAGES`. If `PRED` is unset, presets whose YOLO folder name contains `_exp` (EXP-001 / EXP-002) pick the **newest** `predictions_val.json` under `experiments/yolo/<run>*` so Ultralytics-incremented dirs (e.g. `…2`) win over stale canonical paths.

---

## 🧠 Experiments

### EXP-001: Small object filtering

* During preparation, remove **train**-split GT instances below a configurable area (or side) threshold; **val and test COCO JSON match EXP-000** (same seed/split/raw data), so val mAP compares fairly.
* Config: [`configs/exp001_prepare.yaml`](../configs/exp001_prepare.yaml) sets `filter.apply_to: train` and `filter.min_area_px: 1024` (COCO “small” is &lt; 32²). Override with Hydra, e.g. `filter.min_area_px=2048`.
* Outputs: `datasets/processed/test_run_exp001/`, `experiments/yolo/test_run_exp001/`, `experiments/results/test_run_exp001_metrics.json`.

**Prerequisite:** run EXP-000 first so `experiments/results/test_run_metrics.json` exists (for automatic comparison at the end of the EXP-001 script).

```bash
chmod +x scripts/run_exp001.sh   # once
./scripts/run_exp001.sh
```

This prepares the filtered dataset, trains 1 epoch (`yolo26n.pt`, same hyperparameters as smoke test), runs val inference to `experiments/yolo/test_run_exp001/predictions_val.json`, evaluates to `experiments/results/test_run_exp001_metrics.json`, and if the baseline file exists writes **`experiments/results/exp001_vs_baseline.json`** and prints a short delta summary. Device: `EXP001_DEVICE` or `SMOKE_DEVICE` (same semantics as EXP-000).

**Compare only** (if metrics files already exist):

```bash
python scripts/evaluation/compare_metrics.py \
  --baseline experiments/results/test_run_metrics.json \
  --compare experiments/results/test_run_exp001_metrics.json \
  --out experiments/results/exp001_vs_baseline.json \
  --summary-label "EXP-001 vs baseline" \
  --evaluation-note "EXP-001 train-only filter; val GT matches EXP-000."
```

**Visualizations** (predictions + comparisons; no retrain):

```bash
VIZ_PRESET=exp001 ./scripts/run_visualization.sh
# or: ./scripts/run_visualization.sh exp001
```

Writes under `experiments/visualizations/test_run_exp001/`.

**Note:** Global filtering (`filter.apply_to: all` in [`prepare_dataset.yaml`](../configs/prepare_dataset.yaml)) still filters every split; EXP-001 uses `apply_to: train` only.

---

### EXP-002: Higher resolution

* **Same prepared data as EXP-000** (`datasets/processed/test_run`): no extra filtering or second prepare output; val COCO GT and val images are identical to the baseline run.
* **Single change:** Ultralytics training/inference image size **1280** vs **320** in [`scripts/run_smoke_test.sh`](../scripts/run_smoke_test.sh) (everything else matches smoke: `yolo26n.pt`, 1 epoch, batch 4, workers 0).
* Training config: [`configs/train/yolo_exp002.yaml`](../configs/train/yolo_exp002.yaml) (compose with `python scripts/train/train_yolo.py --config-name=train/yolo_exp002`).
* Outputs: `experiments/yolo/test_run_exp002/`, `experiments/results/test_run_exp002_metrics.json`.

**Prerequisite:** run EXP-000 first (`./scripts/run_smoke_test.sh`) so `datasets/processed/test_run` and `experiments/results/test_run_metrics.json` exist.

```bash
chmod +x scripts/run_exp002.sh   # once
./scripts/run_exp002.sh
```

This trains at `imgsz=1280`, runs val inference to `experiments/yolo/test_run_exp002/predictions_val.json`, evaluates to `experiments/results/test_run_exp002_metrics.json`, and if the baseline metrics file exists writes **`experiments/results/exp002_vs_baseline.json`** (includes **mAP / mAP_small** and **FPS / latency** deltas via `compare_metrics.py`). Device: `EXP002_DEVICE` or `SMOKE_DEVICE`. If GPU memory is insufficient at batch 4, run `EXP002_BATCH=2 ./scripts/run_exp002.sh` and document the batch change (strict hyperparam identity no longer holds).

**Compare only** (if metrics files already exist):

```bash
python scripts/evaluation/compare_metrics.py \
  --baseline experiments/results/test_run_metrics.json \
  --compare experiments/results/test_run_exp002_metrics.json \
  --out experiments/results/exp002_vs_baseline.json \
  --summary-label "EXP-002 vs baseline" \
  --evaluation-note "EXP-002: same val GT as EXP-000; higher imgsz only."
```

**Visualizations** (no retrain):

```bash
VIZ_PRESET=exp002 ./scripts/run_visualization.sh
# or: ./scripts/run_visualization.sh exp002
```

Writes under `experiments/visualizations/test_run_exp002/` (val overlays use `datasets/processed/test_run` for GT/images).

---

### EXP-002b: Resolution sweep

* **Same prepared data and val GT as EXP-000** (`datasets/processed/test_run`): no filtering; identical validation COCO JSON and images.
* **Variable:** Ultralytics `imgsz` only — sweep **640, 768, 896, 1024** (same model `yolo26n.pt`, 1 epoch, batch 4, workers 0 as smoke-style runs unless overridden).
* **Training config:** [`configs/train/yolo_exp002b.yaml`](../configs/train/yolo_exp002b.yaml); per-size run dirs `experiments/yolo/exp002b_imgsz{640|768|896|1024}/`.
* **Inference / evaluation:** [`scripts/inference/infer_yolo.py`](../scripts/inference/infer_yolo.py) and [`scripts/evaluation/evaluate.py`](../scripts/evaluation/evaluate.py) take `--imgsz` so predict and FPS benchmark match each trained resolution.

**Reuse 640 from baseline:** If `experiments/yolo/test_run/config.yaml` has `imgsz: 640` and both `experiments/results/test_run_metrics.json` and `experiments/yolo/test_run/predictions_val.json` exist, the sweep **copies** baseline metrics to `experiments/results/exp002b_imgsz640_metrics.json` and skips train/infer/eval for 640. (Current smoke baseline uses `imgsz=320`, so all four sizes are normally trained.)

```bash
chmod +x scripts/run_exp002b.sh   # once
./scripts/run_exp002b.sh
```

**Outputs:**

* Per size: `experiments/results/exp002b_imgsz*_metrics.json`
* Aggregated: `experiments/results/exp002b_resolution_sweep.json` (compact `summary` table + `runs` metadata)
* Narrative: `experiments/results/exp002b_recommendation.md`
* Plots (matplotlib): `experiments/results/plots/exp002b_mapsmall_vs_imgsz.png`, `exp002b_map_vs_imgsz.png`, `exp002b_fps_vs_imgsz.png`

**Env:** `EXP002B_DEVICE` or `SMOKE_DEVICE`; `EXP002B_BATCH` if VRAM is tight at larger `imgsz`.

**Summarize only** (if metrics files already exist):

```bash
python scripts/evaluation/summarize_resolution_sweep.py --cwd . \
  --out experiments/results/exp002b_resolution_sweep.json \
  --recommendation-out experiments/results/exp002b_recommendation.md \
  --plots-dir experiments/results/plots
```

---

### EXP-003: SAHI sliced inference (no retraining)

* **Same val COCO GT and images as EXP-000** (`datasets/processed/test_run`): no dataset changes.
* **Variable:** [SAHI](https://obss.github.io/sahi/) tiled inference only — compare against **vanilla** `infer_yolo.py` metrics for the same weights.
* **Weights:** `experiments/yolo/test_run/weights/best.pt` (320-train smoke baseline; confirm actual `imgsz` in `experiments/yolo/test_run/config.yaml`) and `experiments/yolo/exp002b_imgsz896/weights/best.pt` (requires a completed EXP-002b run).
* **SAHI parameters:** [`configs/exp003_sahi.yaml`](../configs/exp003_sahi.yaml) (slice size, overlap, confidence; CLI overrides in [`scripts/inference/infer_sahi_yolo.py`](../scripts/inference/infer_sahi_yolo.py)). Each run folder stores a resolved `sahi_config.json` next to `predictions_val.json`.

**Prerequisites:** `pip install -r requirements.txt` (includes `sahi`), EXP-000 artifacts, **`experiments/results/test_run_metrics.json`**, and **`experiments/results/exp002b_imgsz896_metrics.json`** plus 896 weights (run `./scripts/run_exp002b.sh` first if missing).

```bash
chmod +x scripts/run_exp003.sh   # once
./scripts/run_exp003.sh
```

**Env:** `EXP003_DEVICE` or `SMOKE_DEVICE`; optional `EXP003_SAHI_CONFIG` to point at a different YAML.

**Outputs:**

* Predictions + logged SAHI config: `experiments/yolo/test_run_exp003_sahi_base/`, `experiments/yolo/test_run_exp003_sahi_896/`
* Metrics: `experiments/results/test_run_exp003_sahi_base_metrics.json`, `test_run_exp003_sahi_896_metrics.json` (FPS/latency use SAHI when `--sahi-config` is passed to `evaluate.py`)
* Comparisons: `experiments/results/exp003_sahi_vs_baseline.json`, `exp003_sahi_vs_exp002b_896.json`
* Overlays: `experiments/visualizations/test_run_exp003_sahi_base/`, `…/test_run_exp003_sahi_896/`
* Narrative: `experiments/results/exp003_sahi_summary.md` (fill numbers after the script run)

**Make:** `make reproduce-exp003` runs the same shell script.

---

### EXP-A000: Ant MOT → YOLO baseline (domain dataset)

* **Separate from COCO128:** indoor/outdoor **ant** tracking frames (e.g. 1920×1080) with **MOT** `gt/gt.txt` (`frame, id, x, y, w, h, …`) and frames under `img1/` (or `img/`) per sequence.
* **Preparation:** [`scripts/datasets/prepare_ants_mot.py`](../scripts/datasets/prepare_ants_mot.py) + [`configs/datasets/ants_mot_prepare.yaml`](../configs/datasets/ants_mot_prepare.yaml) — **temporal split** (`per_sequence` default: first ~80% of frames per sequence → train, last ~20% → val; no random split).
* **Outputs:** `datasets/ants_yolo/` (YOLO layout, `dataset.yaml`, `annotations/instances_{train,val}.json`, `analysis.json`, `prepare_manifest.json`). Directory is **gitignored**; regenerate locally.
* **Canonical full baseline (20 epochs):** [`configs/train/yolo_ants_expA000_full.yaml`](../configs/train/yolo_ants_expA000_full.yaml) → `experiments/yolo/ants_expA000_full/`, metrics `experiments/results/ants_expA000_full_metrics.json`, relative-area stats `experiments/results/ants_expA000_relative_metrics.json`, overlays `experiments/visualizations/ants_expA000_full/`, report `experiments/results/ants_expA000_full_summary.md` (includes smoke vs full table when smoke metrics exist). Orchestrator: [`scripts/run_ants_expA000_full.sh`](../scripts/run_ants_expA000_full.sh) (`make reproduce-ants-full`).
* **Smoke (1 epoch, pipeline check):** [`configs/train/yolo_ants_expA000_smoke.yaml`](../configs/train/yolo_ants_expA000_smoke.yaml) → `experiments/yolo/ants_expA000_smoke/`, metrics `experiments/results/ants_expA000_smoke_metrics.json`.
* **Legacy baseline (20 epochs, different artifact names — deprecated for new work):** [`configs/train/yolo_ants_expA000.yaml`](../configs/train/yolo_ants_expA000.yaml) → `experiments/yolo/ants_expA000/`, metrics `experiments/results/ants_expA000_metrics.json`, summary `experiments/results/ants_expA000_summary.md`. Prefer **`run_ants_expA000_full.sh`** so results align with docs and `research_analysis.md`.

**Prerequisite:** set **`ANTS_DATASET_ROOT`** to your `Ant_dataset` root (path with MOT sequences).

```bash
chmod +x scripts/run_ants_prepare.sh scripts/run_ants_expA000_smoke.sh scripts/run_ants_expA000.sh scripts/run_ants_expA000_full.sh   # once
export ANTS_DATASET_ROOT="/path/to/Ant_dataset"
./scripts/run_ants_prepare.sh
./scripts/run_ants_expA000_smoke.sh    # optional 1-epoch pipeline check
./scripts/run_ants_expA000_full.sh     # canonical 20-epoch baseline (relative metrics, viz cap, summary)
# legacy only: ./scripts/run_ants_expA000.sh
```

**Env:** `ANTS_DEVICE` or `SMOKE_DEVICE` for CUDA/CPU. For viz, default overlay cap is **250** val images; set `ANTS_VIZ_MAX_IMAGES=all` to render all, or e.g. `ANTS_VIZ_MAX_IMAGES=1073` for the full val set. Optional Hydra overrides on prepare: `./scripts/run_ants_prepare.sh split_strategy=global_sorted link_mode=copy`.

**GT samples:** `experiments/visualizations/ants_dataset/` (from `viz_ant_gt_samples.py` inside prepare script).

---

### EXP-A002b: Ant resolution sweep (640–1024)

* **Same** `datasets/ants_yolo` temporal split and val COCO GT as EXP-A000 full; **only** training/inference `imgsz` changes (20 epochs per size).
* **Config:** [`configs/train/yolo_ants_expA002b.yaml`](../configs/train/yolo_ants_expA002b.yaml). **Orchestrator:** [`scripts/run_ants_expA002b.sh`](../scripts/run_ants_expA002b.sh) (`make reproduce-ants-expA002b`).
* **Runs:** `experiments/yolo/ants_expA002b_imgsz{640,768,896,1024}/`. **640 reuse:** if `ants_expA000_full` was trained at `imgsz=640`, metrics are **copied** to `ants_expA002b_imgsz640_metrics.json` (no retrain); preds/viz use `ants_expA000_full/predictions_val.json`.
* **Per-run metrics:** `experiments/results/ants_expA002b_imgsz*_metrics.json` (same `evaluate.py` fields as EXP-A000).
* **Aggregate:** `experiments/results/ants_expA002b_resolution_sweep.json` (wrapper with `summary` rows: mAP, mAP50, mAP_medium, P, R, fps, `latency_ms`), `ants_expA002b_recommendation.md` (trade-off uses **mAP_medium** + median FPS rule), plots `experiments/results/plots/ants_expA002b_*.png` via [`summarize_ants_resolution_sweep.py`](../scripts/evaluation/summarize_ants_resolution_sweep.py).
* **Relative areas (all resolutions):** `experiments/results/ants_expA002b_relative_metrics.json` from [`ants_relative_sweep_aggregate.py`](../scripts/evaluation/ants_relative_sweep_aggregate.py).
* **Viz:** `experiments/visualizations/ants_expA002b/imgsz*/` — `comparisons/` highlights FN (orange) / FP (thick red); default overlay cap **250** (`ANTS_VIZ_MAX_IMAGES`).
* **Report:** `experiments/results/ants_expA002b_summary.md` from [`write_ants_expA002b_summary.py`](../scripts/evaluation/write_ants_expA002b_summary.py). Compare to reference [`experiments/results/ants_expA000_full_metrics.json`](../experiments/results/ants_expA000_full_metrics.json).

**Prerequisite:** prepared `datasets/ants_yolo/`; for 640 reuse, completed `ants_expA000_full` at 640.

```bash
chmod +x scripts/run_ants_expA002b.sh   # once
export ANTS_DATASET_ROOT="/path/to/Ant_dataset"   # if not already prepared
./scripts/run_ants_prepare.sh          # if needed
./scripts/run_ants_expA000_full.sh     # optional but enables 640 reuse
# OOM: EXP_A002B_BATCH=2 ./scripts/run_ants_expA002b.sh
./scripts/run_ants_expA002b.sh
```

**Next (not automated here):** EXP-A003 SAHI, EXP-A004 ANTS.

---

### EXP-004: Dataset balancing (future)

* Oversample images with small objects
* Goal: improve recall
* Expected: better detection of rare small objects

---

### EXP-005: Bounding box scaling (future)

* Slightly enlarge small bounding boxes
* Goal: improve learnability
* Expected: better detection stability

---

## 📊 Metrics

Each experiment must report:

* mAP
* mAP@0.5
* mAP_small
* Precision
* Recall
* FPS
* Latency (mean ms per image in `evaluate.py` benchmark; compare via `compare_metrics.py` for A/B)
* For **EXP-003**, FPS/latency are measured over **SAHI sliced** inference per full image when evaluation is run with `--sahi-config` (see [`scripts/evaluation/sahi_bench.py`](../scripts/evaluation/sahi_bench.py))
* For **EXP-A002b** (ants), treat **mAP_medium** as the primary COCO bucket when interpreting the sweep (mAP_small is often −1 on this GT); aggregated JSON uses `latency_ms` (= `evaluate.py` mean latency).

---

## ⚠️ Rules

* Change only ONE variable per experiment
* Use same dataset split
* Use same evaluation pipeline
* Log everything

---

## 🧠 Execution Strategy

1. Run baseline (EXP-000)
2. Validate pipeline
3. Run experiments one-by-one
4. Compare results
