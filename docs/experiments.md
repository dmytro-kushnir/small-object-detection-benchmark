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

Writes under `experiments/visualizations/test_run/`: `predictions/` (GT green, preds red), `comparisons/` (FN / FP emphasis), `dataset/` (random GT samples; train split if prepared, else val). Default preset is EXP-000 (`test_run`). For EXP-001 figures use `VIZ_PRESET=exp001` or `./scripts/run_visualization.sh exp001`. Override paths with `PRED`, `GT`, `IMAGES_DIR`, `VIZ_OUT`, `DATASET_GT`, `DATASET_IMAGES`.

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
  --out experiments/results/exp001_vs_baseline.json
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

* Increase image size (e.g., 640 → 1024)
* Goal: better small object visibility
* Expected: higher mAP_small

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
