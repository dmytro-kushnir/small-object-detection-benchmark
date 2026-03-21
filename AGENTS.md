# 🧠 AI Agent Instructions: Small Object Detection Benchmark

## 🎯 Goal

Build a **reproducible research framework** for benchmarking object detection models
(**YOLOv26**, **RF-DETR**) on **small object detection tasks** across:

* Edge devices (Jetson Nano)
* Desktop GPU (RTX 4070)

The system must support:

* Training
* Inference
* Evaluation
* Experiment tracking

---

## 🧱 Project Architecture

Follow this structure strictly:

```
/configs
/datasets
/models
/scripts
/experiments
/docs
/docker
```

### Key Rules

* Separate **model-specific logic** (YOLO vs DETR)
* Keep **evaluation unified**
* Ensure **reproducibility via configs**
* Avoid hardcoded paths

---

## ⚙️ Environment Setup

Use Python 3.10+

Create:

* `requirements.txt`
* optional `pyproject.toml`

Core dependencies:

* torch
* torchvision
* numpy
* opencv-python
* matplotlib
* pycocotools
* tqdm
* hydra-core (for configs)

---

## 📁 Dataset Handling

Implement:

```
/scripts/datasets/prepare_dataset.py
```

The `datasets/` directory in the repo is often a **symlink** to a larger disk (raw/processed data live there). The preparation script stays under `scripts/datasets/` so it remains version-controlled.

Responsibilities:

* Convert dataset to COCO format
* Support small object filtering (by bbox size)
* Train/val/test split
* Resize / normalize images

---

## 🧠 Model Integration

### YOLOv26

* Wrap inference and training in:

  ```
  /scripts/train/train_yolo.py
  /scripts/inference/infer_yolo.py
  ```
* Use existing YOLO repo if possible
* Avoid modifying original source — wrap it

---

### RF-DETR

* Same structure:

  ```
  /scripts/train/train_detr.py
  /scripts/inference/infer_detr.py
  ```

---

## 🧪 Training Pipeline

Each training script must:

* Accept config file
* Save:

  * weights
  * logs
  * metrics
* Use consistent output structure:

```
/experiments/{model}/{run_id}/
```

---

## 🔍 Inference Pipeline

* Input: image / folder
* Output: JSON (COCO predictions format)

---

## 📊 Evaluation

Implement unified evaluator:

```
/scripts/evaluation/evaluate.py
```

Metrics:

* mAP (COCO)
* Precision / Recall
* FPS
* Latency

---

## 📈 Experiment Tracking

Store results in:

```
/experiments/results/
```

Each run must include:

* config.yaml
* metrics.json
* system info (GPU, device)

---

## 🔁 Reproducibility Rules

* All experiments must run via config
* No hidden parameters
* Fix random seeds
* Log everything needed to reproduce

---

## 🐳 Docker Support

Provide:

```
/docker/Dockerfile
```

Must:

* Install dependencies
* Support GPU (if available)
* Run training scripts

---

## 🧪 Development Workflow

When adding new feature:

1. Create script in `/scripts`
2. Add config in `/configs`
3. Ensure evaluation compatibility
4. Document in `/docs`

---

## 📚 Documentation

Maintain:

* `/docs/methodology.md`
* `/docs/datasets.md`
* `/docs/results.md`

---

## 🚫 Anti-Patterns (DO NOT DO)

* ❌ Hardcoded paths
* ❌ Mixing YOLO and DETR logic
* ❌ Manual experiment tracking
* ❌ One-off scripts without config
* ❌ Non-reproducible results

---

## ✅ Success Criteria

* One command to train each model
* One command to evaluate results
* Same dataset used across models
* Results are comparable and reproducible

---

## 🧠 Future Extensions

Design system to support:

* Additional models (RT-DETR, YOLOvX)
* Quantization (TensorRT)
* Jetson deployment optimization
* Real-time pipelines

---

## 🤖 Agent Behavior Guidelines

When generating code:

* Prefer modular design
* Use clear function boundaries
* Add docstrings
* Keep scripts runnable independently
* Avoid overengineering

---

## 🏁 First Task for Agent

1. Create full folder structure
2. Generate:

   * `requirements.txt`
   * dataset preparation script
   * basic evaluation script (mock)
3. Create minimal working pipeline:

   * load image
   * run dummy detection
   * output JSON

Then iterate.

---
