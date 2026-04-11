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

## 🧭 Research Progression (why we started simple)

1. **Pipeline sanity (EXP-000 smoke):** verify end-to-end reproducibility (prepare → train → infer → evaluate) with minimal cost.
2. **Image-size sensitivity (EXP-002 / EXP-002b):** test whether higher `imgsz` helps small-object AP and quantify FPS/latency trade-offs.
3. **Data filtering hypothesis (EXP-001):** check whether train-only removal of very small boxes improves robustness.
4. **Inference-time slicing (EXP-003 SAHI):** test tiled inference without retraining to isolate deployment-time effects.
5. **Architecture expansion (later stage):** move from YOLO-first baselines to RF-DETR and then YOLO vs RF-DETR comparisons on ant and Camponotus tracks.

This staged progression is intentional: early lightweight experiments establish trusted evaluation mechanics before running heavier model-comparison studies.

### SAHI — outcome and why later work stays full-image

We **did** run SAHI as an **inference-only** lever (no retrain) on **COCO128** (**EXP-003**) and on **dense ants** (**EXP-A003**, including a multi-config ablation). Recorded outcomes are summarized in [`docs/research_analysis.md`](research_analysis.md) (**EXP-003** and **EXP-A003**): on those setups, **tiling did not beat** the best **vanilla high-`imgsz`** pipeline for headline mAP (and on COCO128, SAHI **did not** improve **mAP_small** vs vanilla 320; on ants, SAHI **underperformed** vanilla 768, with **no** mAP win in the 54-config sweep). **Later experiments** (e.g. **EXP-A005** / **EXP-A006**, **Camponotus** RF-DETR vs YOLO) therefore default to **full-frame inference** at the reported **`imgsz`** for a **cleaner** apples-to-apples architecture comparison and to avoid a second axis (slice geometry, merge rules, benchmark timing paths). SAHI remains available in scripts for **targeted** revisits if a deployment must keep a **low native `imgsz`** model.

### Why `imgsz` is **768** for ants but **896** for *Camponotus* (main compare)

These tasks use **different** resolution choices on purpose:

* **Dense laboratory ants (YOLO baseline → EXP-A005 / A006):** **`imgsz = 768`** is the **recorded optimum** of the ant **EXP-A002b** sweep (YOLO26n, 20 epochs per size on the ant val split): best **mAP@[0.5:0.95]**, **mAP_medium**, **FPS**, and latency in that table. **896 underperformed 768** on ants in that sweep—so we do **not** carry 896 over from *Camponotus* as if it were the ant optimum.

* ***Camponotus* full-export RF-DETR vs YOLO (e.g. EXP-CAMPO-FULL-2511-YOLO8962-RFDETR-EP60):** both models are trained/evaluated at **`imgsz` / `resolution` = 896** so the headline comparison shares one **nominal input size**. Rationale: **mid-high** square resize (multiple of 32) for **1920×1080** frames with **many small workers**, without defaulting to **1024** everywhere; **loosely** in line with the **COCO128** **EXP-002b** observation that **896** was the **mAP / mAP_small peak** in that **1-epoch toy sweep** (different data—see caveats in [`research_analysis.md`](research_analysis.md)). *Camponotus* **640 vs 896** ablations on **sequence-safe** splits show **schedule-dependent** val/test behavior; **896 is not claimed globally optimal**—it is the **documented operating point** for the published compare JSONs and the Space default.

### *Camponotus* splits: **track_id–majority** (headline compare) *vs* **sequence-safe**

Both are **valid**; they answer **different** leakage questions:

* **Sequence-safe** assigns **whole source clips/sequences** to a single split so **adjacent frames from the same recording** do not straddle train *vs* val/test. Use this when the scientific claim is **clip-level** or **sequence-level** generalization (see **EXP-CAMPO-IDEA1-SEQUENCE-SAFE** in [`research_analysis.md`](research_analysis.md)).

* **Track_id–majority** builds a manifest by placing each CVAT **`track_id`** predominantly into **one** split (by majority of that id’s annotations). It is **not** a guarantee of zero temporal leakage: the logged QA still finds **partial cross-split overlap** for some tracks (e.g. **40 / 638** overlapping ids in one full-export QA file—see [`experiments.md`](experiments.md) Camponotus workflow + `qa_track_id_overlap_in_splits.py`). It aligns with **identity-centric** heuristics and was the manifest wired first into the **full CVAT bundle → `prepare_camponotus_detection_dataset.py` → long YOLO + RF-DETR** runs that produced the **2511-image** headline metrics and the **Hugging Face** demo paths.

**Why the main paper-style table uses track-majority:** the **published** YOLO/RF-DETR compare JSONs (**`camponotus_rfdetr_ep60es_vs_yolo8962_*`**) and **Space** defaults are **pinned** to the **`…_trackidmajor`** export. **Sequence-safe** metrics remain the **conservative** readout for “what if we forbid clip leakage?”—different image counts and schedules; **do not** merge tables without matching manifests.

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

* `scripts/datasets/prepare_dataset.py` (COCO → YOLO / processed splits)
* `scripts/datasets/prepare_ants_mot.py` (optional MOT ant sequences → `datasets/ants_yolo/`)

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
