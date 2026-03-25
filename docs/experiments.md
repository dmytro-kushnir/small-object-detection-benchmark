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

---

### EXP-A003: Ants SAHI vs vanilla imgsz=768 (no retrain)

* **Same** val COCO GT and images as EXP-A000 / EXP-A002b; **no training** — uses weights from **`experiments/yolo/ants_expA002b_imgsz768`** (`best.pt` or `last.pt`).
* **SAHI defaults:** [`configs/expA003_ants_sahi.yaml`](../configs/expA003_ants_sahi.yaml) (512×512 slices, overlap 0.25, `yolo_imgsz: 768`). Override path with **`EXP_A003_SAHI_CONFIG`**.
* **Orchestrator:** [`scripts/run_ants_expA003.sh`](../scripts/run_ants_expA003.sh) (`make reproduce-ants-expA003`).
* **Outputs:** preds `experiments/yolo/ants_expA003_sahi/predictions_val.json`; metrics `experiments/results/ants_expA003_sahi_metrics.json`; compare `experiments/results/ants_expA003_vs_768.json` (baseline: `ants_expA002b_imgsz768_metrics.json`); viz `experiments/visualizations/ants_expA003_sahi/`; report `experiments/results/ants_expA003_summary.md` from [`write_ants_expA003_summary.py`](../scripts/evaluation/write_ants_expA003_summary.py).
* **Fair comparison:** `evaluation_note` in the compare JSON — vanilla 768 uses Ultralytics `predict` timing in baseline metrics; SAHI uses `evaluate.py` with `--sahi-config` (sliced path per full image). Interpret FPS/latency deltas cautiously.

**Prerequisite:** prepared `datasets/ants_yolo/`; completed EXP-A002b **768** run (weights + `ants_expA002b_imgsz768_metrics.json`).

```bash
chmod +x scripts/run_ants_expA003.sh   # once
export ANTS_DATASET_ROOT="/path/to/Ant_dataset"   # if not already prepared
./scripts/run_ants_prepare.sh          # if needed
./scripts/run_ants_expA002b.sh         # if 768 weights/metrics missing
./scripts/run_ants_expA003.sh
```

* **SAHI merge / tiling ablation (optional):** [`scripts/evaluation/run_ants_expA003_sahi_ablation.py`](../scripts/evaluation/run_ants_expA003_sahi_ablation.py) — 54 configs (`perform_standard_pred` × slice size × overlap × `confidence_threshold`) on the same val split and **`ants_expA002b_imgsz768`** weights; writes [`experiments/results/ants_expA003_sahi_ablation.json`](../experiments/results/ants_expA003_sahi_ablation.json) and [`ants_expA003_sahi_ablation_summary.md`](../experiments/results/ants_expA003_sahi_ablation_summary.md). Uses `evaluate.py --skip-inference-benchmark` (COCO + matched P/R only). Temp files under `experiments/yolo/ants_expA003_ablation_scratch/` (gitignored with `experiments/yolo/`).

```bash
chmod +x scripts/run_ants_expA003_sahi_ablation.sh   # once
./scripts/run_ants_expA003_sahi_ablation.sh
# or: make reproduce-ants-expA003-ablation
# smoke: python3 scripts/evaluation/run_ants_expA003_sahi_ablation.py --max-runs 2
# optional early exit: --early-stop-consecutive 12
```

---

### EXP-A004: ANTS v1 — dense-region refinement (no retrain)

* **Same** val COCO GT and images as EXP-A002b / A003; **no training** — weights from **`experiments/yolo/ants_expA002b_imgsz768`**.
* **Pipeline:** stage-1 full-frame YOLO @ `base_imgsz` → dense ROIs (grid default; optional DBSCAN needs commented optional dep in `requirements.txt`) → per-ROI refine @ `refine_imgsz` → merge/NMS → COCO preds JSON.
* **Config:** [`configs/expA004_ants_v1.yaml`](../configs/expA004_ants_v1.yaml). **Orchestrator:** [`scripts/run_ants_expA004.sh`](../scripts/run_ants_expA004.sh) (`make reproduce-ants-expA004`).
* **Outputs (under `experiments/yolo/ants_expA004/`):** `predictions_val.json`, `predictions_stage1_val.json`, `rois.json`, `inference_benchmark.json` (from [`bench_ants_v1.py`](../scripts/evaluation/bench_ants_v1.py)), copied `config.yaml`, `system_info.json`.
* **Metrics:** `experiments/results/ants_expA004_ants_metrics.json` — `evaluate.py` uses **`--inference-benchmark-json`** so FPS/latency reflect the **full ANTS** path, not a single vanilla `predict`.
* **Compare:** `experiments/results/ants_expA004_vs_baseline.json` ([`compare_ants_expA004.py`](../scripts/evaluation/compare_ants_expA004.py)) — deltas vs **`ants_expA002b_imgsz768_metrics.json`** and vs **`ants_expA003_sahi_metrics.json`** when that file exists.
* **Viz:** `experiments/visualizations/ants_expA004/` — `baseline_comparisons/` (if `ants_expA002b_imgsz768/predictions_val.json` exists), `ants_comparisons/`, `rois_debug/` ([`viz_ants_expA004_comparisons.py`](../scripts/visualization/viz_ants_expA004_comparisons.py), [`viz_ants_rois.py`](../scripts/visualization/viz_ants_rois.py)).
* **Report:** `experiments/results/ants_expA004_summary.md` from [`write_ants_expA004_summary.py`](../scripts/evaluation/write_ants_expA004_summary.py).

**Prerequisite:** prepared `datasets/ants_yolo/`; EXP-A002b **768** weights + `ants_expA002b_imgsz768_metrics.json`. Optional: EXP-A003 SAHI metrics for the three-way compare.

```bash
chmod +x scripts/run_ants_expA004.sh   # once
./scripts/run_ants_expA002b.sh       # if 768 bundle missing
# optional: ./scripts/run_ants_expA003.sh  # for SAHI path in compare JSON
./scripts/run_ants_expA004.sh
```

**Smoke / subset:** `EXP_A004_MAX_IMAGES=5 ./scripts/run_ants_expA004.sh` (caps infer + bench only; evaluation still runs on the subset preds).

**Config (ANTS v1 YAML):** `enable_dense_rois` (skip ROI/refine when false); `enable_post_merge_nms` (when false, no extra torchvision NMS after merge); `pipeline_mode`: `merged` | `stage1_only` | `union_refined`; optional `predict_iou`; `refine_min_score`; `nms_iou`; `nms_class_agnostic` (`batched_nms` per class when false).

**Orchestrator env:** `EXP_A004_METRICS_OUT` (default `experiments/results/ants_expA004_ants_metrics.json`), `EXP_A004_COMPARE_OUT`, `EXP_A004_EXPERIMENT_ID`. **Fixed run bundle:** `make reproduce-ants-expA004-fixed` → `ants_expA004_fixed_metrics.json` + `ants_expA004_fixed_vs_baseline.json`.

**Debug / parity:**

- [`scripts/inference/debug_ants_baseline_parity.py`](../scripts/inference/debug_ants_baseline_parity.py) — ANTS with dense ROIs off vs `experiments/yolo/ants_expA002b_imgsz768/predictions_val.json` (per-image box lists; `--tol` for float slack). Requires **PyTorch + Ultralytics** (activate the same venv you use for YOLO train/infer).
- [`scripts/inference/debug_ants_merge_roundtrip.py`](../scripts/inference/debug_ants_merge_roundtrip.py) — COCO preds through merge with empty refined (should match pre-merge list modulo clamping). **Checked:** full val **1073** images on baseline `predictions_val.json` → **0** mismatches (no Ultralytics needed — COCO + merge path only).
- [`infer_ants_v1.py`](../scripts/inference/infer_ants_v1.py): `--parity-baseline`, `--pipeline-mode`, `--dump-refine-viz DIR`, `--max-refine-viz-rois` (ROI crops + green boxes in **crop** coordinates under e.g. `experiments/visualizations/ants_expA004/debug_refine/`).
- [`scripts/run_ants_expA004_staged_eval.sh`](../scripts/run_ants_expA004_staged_eval.sh) — four-stage val metrics (4× infer + `evaluate.py --skip-inference-benchmark`); scratch preds under `experiments/yolo/ants_expA004_staged_scratch/`.

---

### EXP-A005: RF-DETR baseline on ants (vs YOLO26)

* **Same** val COCO GT and split as EXP-A002b (category id **0** = ant). **No** SAHI/ANTS in this experiment.
* **COCO for RF-DETR:** Roboflow layout under `datasets/ants_coco/` — `train/_annotations.coco.json`, `valid/_annotations.coco.json` ( **`valid/`** = same images as `ants_yolo/images/val/`). Build with [`prepare_ants_coco_rfdetr.py`](../scripts/datasets/prepare_ants_coco_rfdetr.py) / `./scripts/run_ants_prepare_coco.sh`.
* **Config:** [`configs/expA005_ants_rfdetr.yaml`](../configs/expA005_ants_rfdetr.yaml). **Train:** [`train_rfdetr_ants.py`](../scripts/train/train_rfdetr_ants.py). **Infer:** [`infer_rfdetr.py`](../scripts/inference/infer_rfdetr.py). **Bench:** [`bench_rfdetr.py`](../scripts/evaluation/bench_rfdetr.py).
* **Artifacts:** `experiments/rfdetr/ants_expA005/` — `weights/best.pth`, `predictions_val.json`, `inference_benchmark.json`, `config.yaml`, `system_info.json`.
* **Metrics:** `experiments/results/ants_expA005_rfdetr_metrics.json` — `evaluate.py` with **`--inference-benchmark-json`** (RF-DETR timed path).
* **Compare:** `experiments/results/ants_expA005_rfdetr_vs_yolo.json` ([`compare_ants_expA005.py`](../scripts/evaluation/compare_ants_expA005.py)) vs **`ants_expA002b_imgsz768_metrics.json`**.
* **Viz:** `experiments/visualizations/ants_expA005_rfdetr/` — YOLO panels, RF-DETR panels, `side_by_side/` ([`viz_ants_expA005_comparisons.py`](../scripts/visualization/viz_ants_expA005_comparisons.py)).
* **Report:** `experiments/results/ants_expA005_rfdetr_summary.md` ([`write_ants_expA005_summary.py`](../scripts/evaluation/write_ants_expA005_summary.py)).

**Prerequisite:** `pip install rfdetr` (or `pip install -e .[rfdetr]`); prepared `datasets/ants_yolo/`; EXP-A002b **768** weights + `predictions_val.json` + metrics for compare/viz.

```bash
chmod +x scripts/run_ants_expA005.sh   # once
./scripts/run_ants_prepare_coco.sh    # if datasets/ants_coco missing
# optional: EXP_A005_SKIP_TRAIN=1 if weights already in experiments/rfdetr/ants_expA005/weights/best.pth
./scripts/run_ants_expA005.sh
# or: make reproduce-ants-expA005
```

**Smoke:** `EXP_A005_MAX_IMAGES=20 ./scripts/run_ants_expA005.sh` (caps infer + bench only).

**Env:** `EXP_A005_CONFIG`, `RFDETR_DEVICE`, `EXP_A005_SKIP_TRAIN`, `EXP_A005_METRICS_OUT`, `EXP_A005_YOLO_METRICS`, `EXP_A005_VIZ_MAX_IMAGES`, etc. (see [`run_ants_expA005.sh`](../scripts/run_ants_expA005.sh)).

---

### EXP-A006: RF-DETR + ByteTrack + temporal smoothing (ants)

* Detector input is EXP-A005 optimized RF-DETR predictions (`predictions_val_optinfer.json`) by default; if missing, EXP-A006 reruns optimized inference.
* Tracking uses [`track_rfdetr_bytetrack.py`](../scripts/inference/track_rfdetr_bytetrack.py) with `supervision.ByteTrack`; sequence reset uses manifest `sequence_map` when available, otherwise single-sequence fallback.
* Smoothing uses [`smooth_tracks_expA006.py`](../scripts/inference/smooth_tracks_expA006.py): remove tracks shorter than 3 frames, fill 1-frame gaps (linear bbox interpolation), replace confidence with track-average score.
* Pipeline benchmark combines detector latency + tracking/smoothing overhead via [`bench_expA006_tracking.py`](../scripts/evaluation/bench_expA006_tracking.py).
* Outputs:
  * `experiments/rfdetr/ants_expA006_tracks.json`
  * `experiments/rfdetr/ants_expA006_smoothed_predictions.json`
  * `experiments/results/ants_expA006_tracking_metrics.json`
  * `experiments/results/ants_expA006_vs_baseline.json`
  * `experiments/visualizations/ants_expA006_tracking/`
  * `experiments/results/ants_expA006_summary.md`

```bash
chmod +x scripts/run_ants_expA006.sh   # once
./scripts/run_ants_expA006.sh
# or: make reproduce-ants-expA006
```

**Config:** [`configs/expA006_ants_tracking.yaml`](../configs/expA006_ants_tracking.yaml)

---

### Camponotus Dataset Workflow (data prep + labeling, no training)

Goal: prepare a two-class detection dataset (`ant`, `trophallaxis`) from in-situ and external sources, ready for later YOLO/RF-DETR experiments.

Core assets:

* Labeling policy: [`camponotus_labeling_guidelines.md`](camponotus_labeling_guidelines.md)
* CVAT workflow: [`camponotus_cvat_workflow.md`](camponotus_cvat_workflow.md)
* Workflow config: [`configs/datasets/camponotus_workflow.yaml`](../configs/datasets/camponotus_workflow.yaml)
* Orchestrator: [`scripts/run_camponotus_dataset_workflow.sh`](../scripts/run_camponotus_dataset_workflow.sh)
* Key scripts:
  * frame extraction: [`extract_camponotus_frames.py`](../scripts/datasets/extract_camponotus_frames.py)
  * prelabels: [`bootstrap_camponotus_autolabel.py`](../scripts/datasets/bootstrap_camponotus_autolabel.py)
  * split manifest: [`split_camponotus_dataset.py`](../scripts/datasets/split_camponotus_dataset.py)
  * conversion/export: [`prepare_camponotus_detection_dataset.py`](../scripts/datasets/prepare_camponotus_detection_dataset.py)
  * analysis: [`analyze_camponotus_dataset.py`](../scripts/datasets/analyze_camponotus_dataset.py)
  * validation: [`validate_camponotus_dataset.py`](../scripts/datasets/validate_camponotus_dataset.py)
  * sample viz: [`viz_camponotus_dataset_samples.py`](../scripts/visualization/viz_camponotus_dataset_samples.py)
  * RF-DETR Roboflow export: [`prepare_camponotus_coco_rfdetr.py`](../scripts/datasets/prepare_camponotus_coco_rfdetr.py) (optional: `CAMPO_PREP_RFDETR_COCO=1` with the orchestrator above)
  * Idea 2 ant-only derivative: [`export_camponotus_ant_only_for_idea2.py`](../scripts/datasets/export_camponotus_ant_only_for_idea2.py)

Outputs:

* `datasets/camponotus_processed/splits.json`
* `datasets/camponotus_processed/analysis.json`
* `datasets/camponotus_yolo/`
* `datasets/camponotus_coco/annotations/instances_{train,val,test}.json`
* `datasets/camponotus_rfdetr_coco/` (when RF-DETR export is run)
* `experiments/visualizations/camponotus_dataset/`

Research phases (Ideas 1–3): [`camponotus_research_roadmap.md`](camponotus_research_roadmap.md). **YOLO vs RF-DETR (Idea 1):** [`run_camponotus_rfdetr_exp.sh`](../scripts/run_camponotus_rfdetr_exp.sh) + [`configs/expCAMPO_rfdetr.yaml`](../configs/expCAMPO_rfdetr.yaml).

```bash
chmod +x scripts/run_camponotus_dataset_workflow.sh   # once
./scripts/run_camponotus_dataset_workflow.sh
# or: make prepare-camponotus-dataset
```

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
* For **EXP-A003** (ants SAHI), primary readout is **`ants_expA003_vs_768.json`** (Δ vs vanilla 768); see `evaluation_note` for inference-path differences when comparing FPS/latency.
* For **EXP-A004** (ANTS v1), primary readout is **`ants_expA004_vs_baseline.json`** (or **`ants_expA004_fixed_vs_baseline.json`** after `make reproduce-ants-expA004-fixed`); throughput in the metrics JSON comes from **`bench_ants_v1.py`** via **`evaluate.py --inference-benchmark-json`** (full two-stage path).
* For **EXP-A005** (ants RF-DETR), primary readout is **`ants_expA005_rfdetr_vs_yolo.json`**; throughput from **`bench_rfdetr.py`** via **`evaluate.py --inference-benchmark-json`**. Resolution/train budget may differ from YOLO `imgsz=768` — see `evaluation_note` in the compare JSON.
* For **EXP-A006** (RF-DETR temporal), primary readout is **`ants_expA006_vs_baseline.json`**; throughput from combined detector+temporal benchmark JSON (`bench_expA006_tracking.py`) passed through `evaluate.py --inference-benchmark-json`.

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
