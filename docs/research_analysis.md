# Research analysis (living document)

Narrative synthesis for a future paper. **Update this file after each experiment** with new rows, figures, and takeaways. Raw metrics stay in `experiments/results/*.json`; this file is for interpretation and reporting.

- Procedures: [`experiments.md`](experiments.md), [`reproduction.md`](reproduction.md)
- Compare two metric JSONs: `python scripts/evaluation/compare_metrics.py --baseline … --compare … --out …`

---

## Shared setup (EXP-000 / EXP-001 as currently scripted)

| Item | Value |
|------|--------|
| Model | YOLO26n (`yolo26n.pt`) |
| Training | 1 epoch, batch 4, `imgsz=320`, workers 0 (smoke / EXP-001 scripts) |
| Data | COCO128 via `scripts/datasets/download_coco128.py`; **same seed and split** for both runs (`seed` from `prepare_dataset.yaml`; split ratios from shared [`configs/coco128_exp_split.yaml`](../configs/coco128_exp_split.yaml), not the generic `0.8/0.1/0.1` default in `prepare_dataset.yaml`) |
| Evaluation | Same **validation** COCO GT (same image set and instances) for EXP-000 and EXP-001: `filter.apply_to: train` only alters **train** YOLO labels; val COCO JSON matches the baseline run. |
| Primary metrics | COCO mAP / AR from `scripts/evaluation/evaluate.py`; deltas from `compare_metrics.py` |

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

## Changelog

| Date | Experiment(s) | Summary |
|------|----------------|---------|
| 2026-03-21 | EXP-000 vs EXP-001 | Initial write-up; train-only small-box filter; no mAP_small gain, overall metrics slightly worse on recorded run. |

*(Append new rows as you add EXP-002, …)*
