# CLI commands (common)

Short **copy-paste** patterns for Camponotus Idea 1 and unified evaluation. Run from the **repository root**; adjust dataset names, paths, and `WEIGHTS` to your run.

For full experiment narratives and numbers, see [`research_analysis.md`](research_analysis.md). For methodology and configs, see [`methodology.md`](methodology.md).

---

## Camponotus — RF-DETR Roboflow-style dataset

Builds `train/` + `valid/` + mirrored annotations from a Camponotus YOLO export (example: **track_id–majority**):

```bash
python3 scripts/datasets/prepare_camponotus_coco_rfdetr.py \
  --camponotus-yolo-root datasets/camponotus_yolo/camponotus_full_export_unique_trackidmajor \
  --out-root datasets/camponotus_rfdetr_coco_trackidmajor
```

For **sequence-safe**, swap the YOLO root (and use a distinct `--out-root` such as `datasets/camponotus_rfdetr_coco_sequence_safe`). If COCO JSONs are not discoverable automatically, set `--camponotus-coco-annotations-root` to the paired `…/annotations` directory.

End-to-end orchestrator (prepare + train + eval when configured): [`scripts/run_camponotus_rfdetr_exp.sh`](../scripts/run_camponotus_rfdetr_exp.sh) with env overrides as needed.

---

## Camponotus — train RF-DETR

Example: **896** resolution, custom output dir (track_id–majority export):

```bash
python3 scripts/train/train_rfdetr_ants.py \
  --config configs/expCAMPO_rfdetr.yaml \
  --dataset-dir datasets/camponotus_rfdetr_coco_trackidmajor \
  --output-dir experiments/rfdetr/camponotus_rfdetr_trackidmajor_896 \
  --resolution 896 \
  --epochs 30 \
  --batch-size 4 \
  --device cuda:0
```

Weights typically appear under `experiments/rfdetr/<run>/weights/best.pth` (see that run’s `config.yaml`).

**Sequence-safe export + train @896 (fill paper matrix TBD row):**

```bash
python3 scripts/datasets/prepare_camponotus_coco_rfdetr.py \
  --camponotus-yolo-root datasets/camponotus_yolo/camponotus_full_export_unique_sequence_safe \
  --out-root datasets/camponotus_rfdetr_coco_sequence_safe

python3 scripts/train/train_rfdetr_ants.py \
  --config configs/expCAMPO_rfdetr.yaml \
  --dataset-dir datasets/camponotus_rfdetr_coco_sequence_safe \
  --output-dir experiments/rfdetr/camponotus_rfdetr_sequence_safe_896 \
  --resolution 896 \
  --epochs 30 \
  --batch-size 4 \
  --device cuda:0
```

Then mirror the infer/bench/eval pattern below using `…/sequence_safe/…` GT and images and new result filenames; compare to `experiments/results/camponotus_idea1_sequence_safe_full_896_metrics_{val,test}.json`.

---

## RF-DETR — optional faster inference (Camponotus / ants)

[`infer_rfdetr.py`](../scripts/inference/infer_rfdetr.py) calls `model.optimize_for_inference()` when:

```bash
export EXP_A005_OPTIMIZE_INFERENCE=1
```

Re-run `bench_rfdetr.py` + `evaluate.py --inference-benchmark-json` and confirm detection metrics if you cite optimized FPS (ants EXP-A005 documented this pattern).

For **video**, [`track_rfdetr_video.py`](../scripts/inference/track_rfdetr_video.py) supports the same optimization **without** the env var: pass **`--optimize-for-inference`** (optional; faster runtime, verify behavior on a short clip first).

---

## Camponotus — video tracking (RF-DETR)

[`track_rfdetr_video.py`](../scripts/inference/track_rfdetr_video.py) runs RF-DETR + ByteTrack on a single video and writes an MP4 with boxes, IDs, and trails.

**Train/infer resolution (e.g. 896):** there is **no** `--resolution` or `--imgsz` flag. Load the checkpoint that was trained at that resolution (e.g. `…/camponotus_rfdetr_trackidmajor_896/weights/best.pth`); the model’s internal preprocessing matches training.

**Shell:** each `\` line continuation must be the **last character** on the line (no spaces after `\`), or the next line is parsed as a new command.

Example (track_id–majority **896** weights, qualitative / not scored):

```bash
python3 scripts/inference/track_rfdetr_video.py \
  --weights experiments/rfdetr/camponotus_rfdetr_trackidmajor_896/weights/best.pth \
  --source-video /path/to/camponotus_010.mov \
  --out-video experiments/visualizations/camponotus_rfdetr_trackidmajor_896_tracked_camponotus_010.mp4 \
  --model-class RFDETRSmall \
  --conf 0.35 \
  --track-thresh 0.25 \
  --match-thresh 0.8 \
  --track-buffer 30 \
  --trail-len 30 \
  --color-mode state \
  --optimize-for-inference \
  --analytics-out experiments/visualizations/camponotus_rfdetr_trackidmajor_896_tracked_camponotus_010_analytics.json
```

Optional: `--state-priority-soft` and related thresholds (see `--help`) to match prelabel / QA tooling.

---

## Camponotus — video tracking (YOLO)

[`track_yolo_video.py`](../scripts/inference/track_yolo_video.py) uses Ultralytics + BoT-SORT (default) or ByteTrack. Pass **`--imgsz`** when it must match training (e.g. **896** for the Idea 1 896 runs).

```bash
python3 scripts/inference/track_yolo_video.py \
  --weights experiments/yolo/camponotus_idea1_trackidmajor_full_896/weights/best.pt \
  --source-video /path/to/camponotus_010.mov \
  --out-video experiments/visualizations/camponotus_idea1_trackidmajor_full_896_tracked_camponotus_010.mp4 \
  --imgsz 896 \
  --conf 0.35 \
  --tracker botsort \
  --track-thresh 0.25 \
  --match-thresh 0.8 \
  --track-buffer 30 \
  --trail-len 30 \
  --color-mode state \
  --device 0 \
  --analytics-out experiments/visualizations/camponotus_idea1_trackidmajor_full_896_tracked_camponotus_010_analytics.json
```

Adjust `--weights` to your actual run directory. Optional: `--botsort-with-reid`, `--state-priority-soft`.

---

## Camponotus — infer → bench → `evaluate.py`

Use **one** `WEIGHTS` path; repeat for **`val`** and **`test`** by swapping split in `--source`, `--coco-gt`, `--images-dir`, and output filenames.

```bash
WEIGHTS=experiments/rfdetr/camponotus_rfdetr_trackidmajor_896/weights/best.pth

python3 scripts/inference/infer_rfdetr.py \
  --weights "$WEIGHTS" \
  --source datasets/camponotus_yolo/camponotus_full_export_unique_trackidmajor/images/val \
  --coco-gt datasets/camponotus_coco/camponotus_full_export_unique_trackidmajor/annotations/instances_val.json \
  --out experiments/results/camponotus_rfdetr_trackidmajor_896_predictions_val.json \
  --model-class RFDETRSmall \
  --conf 0.25 \
  --class-id-mode multiclass \
  --device 0

python3 scripts/evaluation/bench_rfdetr.py \
  --weights "$WEIGHTS" \
  --source datasets/camponotus_yolo/camponotus_full_export_unique_trackidmajor/images/val \
  --coco-gt datasets/camponotus_coco/camponotus_full_export_unique_trackidmajor/annotations/instances_val.json \
  --model-class RFDETRSmall \
  --conf 0.25 \
  --device 0 \
  --out experiments/results/camponotus_rfdetr_trackidmajor_896_bench_val.json \
  --config configs/expCAMPO_rfdetr.yaml

python3 scripts/evaluation/evaluate.py \
  --gt datasets/camponotus_coco/camponotus_full_export_unique_trackidmajor/annotations/instances_val.json \
  --pred experiments/results/camponotus_rfdetr_trackidmajor_896_predictions_val.json \
  --weights "$WEIGHTS" \
  --images-dir datasets/camponotus_yolo/camponotus_full_export_unique_trackidmajor/images/val \
  --out experiments/results/camponotus_rfdetr_trackidmajor_896_metrics_val.json \
  --experiment-id EXP-CAMPO-RFDETR-TRACKIDMAJORITY-896 \
  --train-config experiments/rfdetr/camponotus_rfdetr_trackidmajor_896/config.yaml \
  --prepare-manifest datasets/camponotus_rfdetr_coco_trackidmajor/camponotus_rfdetr_manifest.json \
  --device 0 \
  --inference-benchmark-json experiments/results/camponotus_rfdetr_trackidmajor_896_bench_val.json
```

**YOLO** Camponotus runs follow the same **GT + predictions + `evaluate.py`** pattern; use `scripts/inference/infer_yolo.py` and pass `--imgsz` when it must match training resolution.

---

## Compare metrics (generic A/B)

Same GT, two recorded `evaluate.py` JSONs:

```bash
python3 scripts/evaluation/compare_metrics.py \
  --baseline experiments/results/<baseline>_metrics_val.json \
  --compare experiments/results/<compare>_metrics_val.json \
  --out experiments/results/<name>_val.json \
  --evaluation-note "Short note for the bundle."
```

---

## Camponotus — RF-DETR vs YOLO

Uses the same delta block as ants EXP-A005; **baseline** is usually a YOLO `experiments/results/*_metrics_*.json`, **compare** is RF-DETR:

```bash
python3 scripts/evaluation/compare_camponotus_rfdetr_vs_yolo.py \
  --baseline experiments/results/camponotus_idea1_trackidmajor_full_896_metrics_val.json \
  --compare experiments/results/camponotus_rfdetr_trackidmajor_896_metrics_val.json \
  --out experiments/results/camponotus_rfdetr_trackidmajor_896_vs_yolo896_val.json
```

---

## Smokes and generic benchmark scripts

| Task | Entry point |
|------|-------------|
| COCO smoke train/eval | `./scripts/run_smoke_test.sh` |
| EXP-001 filtered prepare | `./scripts/run_exp001.sh` |
| Higher `imgsz` (EXP-002) | `./scripts/run_exp002.sh` |
| Visualization | `./scripts/run_visualization.sh` |

---

## Ants (RF-DETR / SAHI) — pointers

Ants workflows use the same **`evaluate.py`** + **`compare_metrics.py`** ideas; see [`experiments.md`](experiments.md) sections **EXP-A002b–A006** and scripts under `scripts/run_ants_*.sh`.
