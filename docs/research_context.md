# Research Context: Small Object Detection Benchmark

## 🎯 Goal

This project benchmarks object detection models for **small object detection performance**.

Models:

* YOLOv26 (Ultralytics)
* RF-DETR (later stage)

The goal is to produce:

* reproducible results
* fair comparison across models
* publishable research outcomes

---

## 🧪 Research Question

How do different object detection architectures perform on small object detection tasks under edge constraints?

---

## 🖥️ Hardware

* Training: RTX 4070
* Inference: Jetson Nano

---

## 📊 Metrics

Primary metrics:

* mAP (IoU=0.5:0.95)
* mAP@0.5
* Precision
* Recall

Additional:

* FPS (inference speed)
* Latency

### Object size categories (COCO standard)

* Small: area < 32²
* Medium: 32²–96²
* Large: > 96²

---

## 🧱 Pipeline

All experiments must follow:

1. Dataset preparation
2. Training
3. Inference
4. Evaluation

Dataset preparation is handled by:

`scripts/datasets/prepare_dataset.py`

This ensures:

* consistent preprocessing
* reproducible splits
* format compatibility (COCO + YOLO)

---

## 🔁 Reproducibility Rules

* All experiments must be config-driven
* No hardcoded paths
* Fixed random seed
* Save for every run:

  * config.yaml
  * metrics.json
  * system info

---

## ⚙️ Current Scope

* Start with YOLO only
* Use small dataset (smoke test)
* Train minimal model (1 epoch)
* Validate full pipeline

Later:

* Add RF-DETR
* Run full experiments

---

## 🚫 Constraints

* Avoid large datasets during development
* Keep experiments lightweight
* Do not optimize prematurely

---

## 🏁 Success Criteria

* End-to-end pipeline works
* Metrics computed correctly
* Results reproducible
* Models comparable
