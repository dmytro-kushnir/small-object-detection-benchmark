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

Writes under `experiments/visualizations/test_run/`: `predictions/` (GT green, preds red), `comparisons/` (FN / FP emphasis), `dataset/` (random GT samples; train split if prepared, else val). Override paths with `PRED`, `GT`, `IMAGES_DIR`, `VIZ_OUT`, `DATASET_GT`, `DATASET_IMAGES`.

---

## 🧠 Experiments

### EXP-001: Small object filtering

* Remove objects below threshold
* Goal: reduce noise
* Expected: higher precision

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
