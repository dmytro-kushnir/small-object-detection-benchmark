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

### EXP-003: Dataset balancing

* Oversample images with small objects
* Goal: improve recall
* Expected: better detection of rare small objects

---

### EXP-004: Bounding box scaling

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
