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

Optional: `--state-priority-soft` and related thresholds (see `--help`) to match prelabel / QA tooling. **`--state-priority-consensus`** (YOLO + RF-DETR) relabels any `normal` detection that overlaps a `trophallaxis` detection at IoU ≥ `--state-priority-iou-thresh` to trophallaxis (no score-gap check; OOD / viz — can add false troph when ants merely touch). Optional **`--temporal-state-window K`** for sliding majority smoothing (see YOLO subsection below).

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

Adjust `--weights` to your actual run directory. Optional: `--botsort-with-reid`, `--state-priority-soft`, `--state-priority-consensus`.

### OOD / qualitative: temporal state smoothing (not COCO benchmark)

After optional `--state-priority-soft`, **`--temporal-state-window K`** (default `0` = off) applies a **sliding majority** on `class_id` per `track_id` over the last `K` frames. Ties keep the current frame’s class. Analytics JSON gains `temporal_state_smooth.window` when `K>0`. Same flag exists on [`track_rfdetr_video.py`](../scripts/inference/track_rfdetr_video.py).

Example (YOLO, match prior OOD diagnostic: conf 0.25 + soft + `K=9`):

```bash
python3 scripts/inference/track_yolo_video.py \
  --weights experiments/yolo/camponotus_idea1_trackidmajor_full_896/weights/best.pt \
  --source-video /path/to/trophalaxis_001_example.mp4 \
  --out-video experiments/visualizations/ood_troph001_k9.mp4 \
  --imgsz 896 \
  --conf 0.25 \
  --state-priority-soft \
  --temporal-state-window 9 \
  --analytics-out experiments/visualizations/ood_troph001_k9_analytics.json
```

Loop `K ∈ {3,5,9,15}` vs `K=0` baseline and compare each `states` and `per_track` in the analytics files. One-shot driver: [`scripts/run_camponotus_ood_temporal_sweep.sh`](../scripts/run_camponotus_ood_temporal_sweep.sh) (`VIDEO=...` required).

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

### External-only inference (no COCO GT)

For ad-hoc PNG/JPEG/WebP not in a benchmark JSON: **omit `--coco-gt`**. Images get sequential `image_id` values starting at **`--synthetic-image-id-start`** (default `1`). Output is still a COCO **list** JSON; `evaluate.py` only scores rows whose `image_id` exists in the GT you pass there.

**YOLO:**

```bash
python3 scripts/inference/infer_yolo.py \
  --weights experiments/yolo/camponotus_idea1_trackidmajor_full_896/weights/best.pt \
  --source /path/to/frame.png \
  --imgsz 896 \
  --out experiments/results/ood_single_frame_yolo.json
```

**RF-DETR:**

```bash
python3 scripts/inference/infer_rfdetr.py \
  --weights experiments/rfdetr/camponotus_rfdetr_trackidmajor_896/weights/best.pth \
  --source /path/to/frame.png \
  --model-class RFDETRSmall \
  --conf 0.25 \
  --class-id-mode multiclass \
  --out experiments/results/ood_single_frame_rfdetr.json
```

Optional overlays (OpenCV, same boxes as JSON): **`--save-vis`** and **`--vis-dir`** (default `…/viz_infer_rfdetr` next to `--out`). Human-readable labels: **`--vis-names-json`** with a JSON object like `{"0":"normal","1":"trophallaxis"}`.

**GT + extra folders/files:** keep `--coco-gt` and add **`--extra-source PATH`** (repeatable). Extra images get ids from **`--extra-image-id-start`** or, by default, **`max(image_id in COCO) + 1`**. Shared helpers: [`infer_image_common.py`](../scripts/inference/infer_image_common.py), [`coco_pred_common.py`](../scripts/inference/coco_pred_common.py) (`max_image_id_in_coco`).

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

## Camponotus — Idea 2 hybrid events (tracking + helper)

Artifact index: [`camponotus_idea2_artifacts.md`](camponotus_idea2_artifacts.md) (what each JSON is, default run folder `experiments/results/idea2/<RUN_NAME>/`).

Inputs:

- MOT JSON from prelabel/tracking export (`bootstrap_camponotus_autolabel.py --mot-out-json`)
- event GT subset (`datasets/camponotus_idea2_event_benchmark_v1.json`)

Run baseline with helper signal:

```bash
python3 scripts/inference/infer_camponotus_idea2_events.py \
  --mot-json experiments/cvat_prelabels/camponotus_prelabels_trackidmajor_mot.json \
  --out experiments/results/camponotus_idea2_events_hybrid_with_helper.json \
  --max-dist-px 90 \
  --pair-score-threshold 0.45 \
  --min-active-frames 12 \
  --max-gap-frames 3
```

Run ablation without helper signal:

```bash
python3 scripts/inference/infer_camponotus_idea2_events.py \
  --mot-json experiments/cvat_prelabels/camponotus_prelabels_trackidmajor_mot.json \
  --out experiments/results/camponotus_idea2_events_hybrid_no_helper.json \
  --disable-helper-signal \
  --max-dist-px 90 \
  --pair-score-threshold 0.45 \
  --min-active-frames 12 \
  --max-gap-frames 3
```

Evaluate each prediction file:

```bash
python3 scripts/evaluation/evaluate_camponotus_idea2_events.py \
  --gt-events datasets/camponotus_idea2_event_benchmark_v1.json \
  --pred-events experiments/results/camponotus_idea2_events_hybrid_with_helper.json \
  --out experiments/results/camponotus_idea2_events_hybrid_with_helper_eval.json \
  --match-tiou-threshold 0.30

python3 scripts/evaluation/evaluate_camponotus_idea2_events.py \
  --gt-events datasets/camponotus_idea2_event_benchmark_v1.json \
  --pred-events experiments/results/camponotus_idea2_events_hybrid_no_helper.json \
  --out experiments/results/camponotus_idea2_events_hybrid_no_helper_eval.json \
  --match-tiou-threshold 0.30
```

Compare helper vs no-helper:

```bash
python3 scripts/evaluation/compare_camponotus_idea2_event_metrics.py \
  --baseline experiments/results/camponotus_idea2_events_hybrid_no_helper_eval.json \
  --compare experiments/results/camponotus_idea2_events_hybrid_with_helper_eval.json \
  --out experiments/results/camponotus_idea2_events_helper_vs_no_helper.json \
  --evaluation-note "Idea 2 hybrid baseline ablation: helper signal contribution."
```

Protocol reference: [`camponotus_idea2_event_protocol.md`](camponotus_idea2_event_protocol.md).

Build a **draft** event GT from existing CVAT COCO `state` + `track_id` labels:

```bash
python3 scripts/datasets/build_camponotus_idea2_event_gt.py \
  --coco-annotations "/media/dmytro/data/datasets/camponotus fellah trophallaxis FULL dataset/annotations/instances_default.json" \
  --out datasets/camponotus_idea2_event_benchmark_v1_auto.json \
  --fps 25 \
  --state-attr state \
  --trophallaxis-state-value trophallaxis \
  --track-id-attr track_id \
  --max-pair-dist-px 90 \
  --min-active-frames 12 \
  --max-gap-frames 3
```

Then review and correct this draft before using it as the frozen benchmark file.

Recommended v1 subset mix (from current Camponotus curation notes):

- almost fully trophallaxis:
  - `seq_camponotus_trophallaxis_007`
  - `seq_camponotus_trophallaxis_005`
  - `seq_camponotus_trophallaxis_003`
- partial:
  - `seq_camponotus_009`
  - `seq_camponotus_007`
- no trophallaxis:
  - `seq_camponotus_003`
  - `seq_camponotus_002`
- ant-dense mostly non-event:
  - `seq_camponotus_010`

Example command using an external-drive CVAT export and explicit sequence allowlist:

```bash
python3 scripts/datasets/build_camponotus_idea2_event_gt.py \
  --coco-annotations "/media/dmytro/data/datasets/camponotus fellah trophallaxis FULL dataset/annotations/instances_default.json" \
  --out datasets/camponotus_idea2_event_benchmark_v1_auto.json \
  --fps 25 \
  --state-attr state \
  --trophallaxis-state-value trophallaxis \
  --track-id-attr track_id \
  --max-pair-dist-px 90 \
  --min-active-frames 12 \
  --max-gap-frames 3 \
  --clip-allowlist "seq_camponotus_trophallaxis_007,seq_camponotus_trophallaxis_005,seq_camponotus_trophallaxis_003,seq_camponotus_009,seq_camponotus_007,seq_camponotus_003,seq_camponotus_002,seq_camponotus_010"
```

One-shot runner (MOT -> events -> eval -> compare) against final GT file:

```bash
GT_EVENTS=datasets/camponotus_idea2_event_benchmark_v1.json \
IN_SITU_ROOT="/media/dmytro/data/datasets/camponotus fellah trophallaxis FULL dataset/images/default/in_situ" \
BACKEND=yolo \
YOLO_WEIGHTS=experiments/yolo/camponotus_idea1_trackidmajor_full_896/weights/best.pt \
RUN_NAME=camponotus_idea2_hybrid_v1_yolo \
OUT_DIR=experiments/results/idea2/camponotus_idea2_hybrid_v1_yolo \
bash scripts/run_camponotus_idea2_hybrid.sh
```

For RF-DETR, switch `BACKEND=rfdetr` and provide `RFDETR_WEIGHTS=/path/to/best.pth`.

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
