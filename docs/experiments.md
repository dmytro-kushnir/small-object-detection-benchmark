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
