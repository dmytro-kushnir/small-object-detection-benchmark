# Camponotus CVAT Workflow

This workflow describes annotation for Camponotus fellah **detection** (two exported classes: `ant` / `trophallaxis`) with optional **instance identity** for future sequence or MOT-style analysis.

Canonical intermediate export for this project is **CVAT COCO 1.0** (bounding boxes), not keypoints.

## 1) Prepare Input Data

Expected raw layout:

- `datasets/camponotus_raw/in_situ/seq_*/` (default: `seq_<sanitized_video_filename>` per clip)
- `datasets/camponotus_raw/external/images/`

For video-derived data, extract frames first using:

- `scripts/datasets/extract_camponotus_frames.py`

Then create `datasets/camponotus_processed/splits.json` with `scripts/datasets/split_camponotus_dataset.py` (default: **stratified** split so trophallaxis-named sequences appear in val/test when possible; use `--no-stratify-trophallaxis` for legacy behavior).

Alternative: `track_id`-majority split manifest (leakage proxy)
- Use `scripts/datasets/split_camponotus_dataset_by_track_id_majority.py` to create a split manifest from `annotations[].attributes.track_id` (majority track_id per image).
- Quantify leakage with `scripts/datasets/qa_track_id_overlap_in_splits.py`.
- Then pass that manifest into `scripts/datasets/prepare_camponotus_detection_dataset.py` via `--split-source manifest --splits <manifest_path>`.

## 2) Create CVAT Task(s)

Recommended setup:

1. Create one task per logical batch (for example by source or sequence groups).
2. Upload images as image lists (preserving sequence ordering for `in_situ`).
3. Define labels (choose **one** pattern):

**A — Recommended (single box label + attributes)**

- Rectangle label: **`ant`** only
- Attributes on that label:
  - **`state`** (mutable): e.g. `normal` vs `trophallaxis`
  - **`track_id`** (number, optional; keep **not** mutable if you want stable identity across frames)

**B — Legacy (two box labels)**

- `ant`
- `trophallaxis`

Tip: Keep `in_situ` and `external` in separate tasks or projects for easier auditing.

## 3) Annotate Bounding Boxes

Follow:

- `docs/camponotus_labeling_guidelines.md`

Critical rule:

- if two ants are in trophallaxis, both ants must end up as exported class `trophallaxis` (`1`) — via **`state`** or via the `trophallaxis` label.

## 4) Interpolation / Tracking Usage

For sequential in-situ frames:

- Use CVAT tracking/interpolation tools to speed up repeated boxes.
- Always manually review keyframes and transitions.
- Re-check **`state`** near behavioral transitions (normal ↔ trophallaxis).
- If you use **`track_id`**, keep it consistent for the same physical ant within a sequence.

For external still images:

- Use standard box annotation (no interpolation needed).

## 5) Export

Export annotations as **COCO 1.0** (detection / bounding boxes) from CVAT and store under:

- `datasets/camponotus_processed/annotations/cvat_coco/`

Recommended naming:

- `cvat_in_situ_batch01.json`
- `cvat_external_batch01.json`
- or merged `cvat_camponotus_all.json`

Keep original exports immutable. Any corrections should produce a new versioned file.

### CVAT `file_name` vs `splits.json`

The split manifest lists repo-relative paths such as `datasets/camponotus_raw/in_situ/seq_camponotus_002/000001.jpg`. Preparation matches each COCO `file_name` using the same key logic (`seq_*/basename`, else basename). **Flat CVAT exports** (e.g. only `000001.jpg`) usually **do not** match those keys, so every image is skipped and splits stay empty.

Use **`--split-source auto`** on `prepare_camponotus_detection_dataset.py`: only images that resolve under `--raw-root` are shuffled and split by ratio (defaults: `ratios` from `splits.json` if present, else 0.7 / 0.15 / remainder test; override with `--train-ratio` / `--val-ratio` / `--auto-split-seed`). Sequence-safe splitting still requires either aligned `file_name` values with **`--split-source manifest`** or running `split_camponotus_dataset.py` on the same directory layout the export uses.

**Sequence-safe workflow (manifest mode):** (1) Put frames under `datasets/camponotus_raw/in_situ/` in **multiple** `seq_*` directories (at least **three** logical sequences so val/test are non-empty — one folder ⇒ the splitter cannot spread frames across splits). Default extraction uses `000001.jpg`-style names **inside each** `seq_*` folder, so the **same basename repeats across sequences**; `align_coco_filenames_to_camponotus_raw.py` will then fail. Prefer `extract_camponotus_frames.py --unique-frame-basenames` (writes `seq_camponotus_001_000001.jpg`, …) so every basename is unique under `raw-root`, then re-run `split_camponotus_dataset.py`. (2) `mkdir -p datasets/camponotus_raw/external/images` (may stay empty). (3) Run `scripts/datasets/split_camponotus_dataset.py` → `datasets/camponotus_processed/splits.json`. (4) If CVAT exports **flat** `file_name` values, run `scripts/datasets/align_coco_filenames_to_camponotus_raw.py --coco … --raw-root datasets/camponotus_raw` so each basename matches **exactly one** file. (5) `prepare_camponotus_detection_dataset.py --split-source manifest --raw-root datasets/camponotus_raw …`.

### Optional `track_id` and `state` in processed exports

Manifest-mode reminder: when you generate a custom split manifest (including the `track_id`-majority one), `prepare_camponotus_detection_dataset.py` should be run with `--split-source manifest --splits <path>` so images are assigned consistently with that manifest.

- `scripts/datasets/prepare_camponotus_detection_dataset.py`:
  - Maps **`attributes.state == trophallaxis`** (configurable via `--state-attr` / `--trophallaxis-state-value`) to exported **`category_id` 1**; any other `state` value → **`category_id` 0**. If `state` is absent on an annotation, it falls back to normalized CVAT category ids (legacy two-label tasks).
  - Writes canonical integer **`track_id`** on each exported COCO annotation when it can be read from the CVAT file: top-level `track_id`, **`attributes[track_id]`** (default attribute name `track_id`), or integer-like `group_id`. Use `--track-id-attr` if your CVAT attribute name differs; use `--strip-track-id` to omit ids.
- `scripts/evaluation/evaluate.py` **ignores** `track_id` for mAP / P@R; it is reserved for future MOT or sequence tooling.
- After export, run `scripts/datasets/validate_camponotus_dataset.py`; use `--strict-track-id` if you require at most one box per `(image_id, track_id)`.

## 6) Prelabels (Semi-Automatic Workflow)

**Pattern A (only `ant` + `state`) vs two-class prelabels:** Bootstrap JSON has classes **ant** and **trophallaxis**. If your CVAT task has **only** rectangle label **`ant`** with attribute **`state`** (`normal` / `trophallaxis`), build a single-label COCO file and **carry behavior into `attributes.state`** (Datumaro/CVAT read per-bbox **`attributes`** on import):

```bash
python3 scripts/datasets/coco_shift_category_ids_for_cvat.py \
  --in datasets/camponotus_processed/prelabels/camponotus_prelabels_coco.json \
  --out datasets/camponotus_processed/prelabels/camponotus_prelabels_coco_cvat_ant_only.json \
  --collapse-to-single-label ant \
  --carry-state-attributes
```

Define **`state`** on label **`ant`** in CVAT with allowed values matching **`--state-normal`** / **`--state-troph`** (defaults: `normal`, `trophallaxis`). If import ignores attributes, fall back to omitting **`--carry-state-attributes`** and set **`state`** manually.

Without **`--carry-state-attributes`**, every box is **`ant`** with no metadata; you set **`state`** while reviewing. **`prepare_camponotus_detection_dataset.py`** maps **`state`** → exported class **1** on export.

**Pattern B (two rectangle labels):** Convert ids only (no collapse): `coco_shift_category_ids_for_cvat.py --in … --out …` (adds **`trophallaxis`** label in CVAT matching the JSON).

If using model-assisted prelabeling:

1. Generate prelabels with `scripts/datasets/bootstrap_camponotus_autolabel.py` (optional **`--cvat-coco-categories`** if you skip the shift script and want 1-based ids in the raw file).
   - For sequential in-situ clips, you can reduce ID flicker and emit stable per-box `track_id` directly in the prelabel COCO via ByteTrack:

```bash
python3 scripts/datasets/bootstrap_camponotus_autolabel.py \
  --images-root datasets/camponotus_raw/in_situ \
  --backend yolo \
  --yolo-weights experiments/yolo/camponotus_yolo26n_v2/weights/best.pt \
  --conf 0.45 \
  --with-tracking \
  --track-thresh 0.25 \
  --match-thresh 0.8 \
  --track-buffer 30 \
  --min-track-len 2 \
  --out datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked.json
```

   - This is useful for Idea 1 annotation stability and serves as a starter for Idea 2 sequence tooling.
   - Optional stronger identity continuity path (YOLO backend only): BoT-SORT with appearance ReID.

```bash
python3 scripts/datasets/bootstrap_camponotus_autolabel.py \
  --images-root datasets/camponotus_raw/in_situ \
  --backend yolo \
  --yolo-weights experiments/yolo/camponotus_yolo26n_v2/weights/best.pt \
  --conf 0.45 \
  --with-tracking \
  --tracker botsort \
  --botsort-with-reid \
  --track-thresh 0.25 \
  --match-thresh 0.8 \
  --track-buffer 30 \
  --min-track-len 2 \
  --out datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_botsort_reid.json
```

   - Compare this output against the ByteTrack output with `scripts/evaluation/compare_camponotus_prelabels_tracking.py` and keep whichever gives fewer short fragmented tracks at acceptable box retention.
   - Optional soft state-priority relabel (no deletion): if a `normal` box strongly overlaps a `trophallaxis` box and score gap is small, relabel `normal -> trophallaxis`.

```bash
python3 scripts/datasets/bootstrap_camponotus_autolabel.py \
  --images-root datasets/camponotus_raw/in_situ \
  --backend yolo \
  --yolo-weights experiments/yolo/camponotus_yolo26n_v2/weights/best.pt \
  --conf 0.45 \
  --with-tracking \
  --tracker botsort \
  --botsort-with-reid \
  --state-priority-soft \
  --state-priority-iou-thresh 0.70 \
  --state-priority-score-gap-max 0.12 \
  --out datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_botsort_reid_soft.json
```
2. For CVAT import, run **`coco_shift_category_ids_for_cvat.py`** (**`--collapse-to-single-label ant --carry-state-attributes`** for pattern A with `state`, or plain shift for pattern B).
3. Import the resulting JSON into CVAT.
4. Correct manually.
5. Export corrected COCO and keep both files:
   - machine prelabels (raw)
   - corrected human-reviewed labels

### Optional: emit MOT-style JSON together with COCO

If you want tracking-native sidecar data from the same run, add `--mot-out-json`.

```bash
python3 scripts/datasets/bootstrap_camponotus_autolabel.py \
  --images-root datasets/camponotus_raw/in_situ \
  --backend yolo \
  --yolo-weights experiments/yolo/camponotus_yolo26n_v2/weights/best.pt \
  --conf 0.45 \
  --with-tracking \
  --tracker botsort \
  --botsort-with-reid \
  --state-priority-soft \
  --state-priority-iou-thresh 0.70 \
  --state-priority-score-gap-max 0.12 \
  --out datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_botsort_reid_soft.json \
  --mot-out-json datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_botsort_reid_soft_mot.json
```

This writes:
- COCO detections (`--out`) with per-annotation `track_id` metadata.
- MOT-style JSON (`--mot-out-json`) with per-sequence frame index and MOTChallenge-like rows.

### Optional: emit CVAT Video 1.1 XML together with COCO

For native CVAT track editing with `state` attribute, add `--cvat-video-xml-out`:

```bash
python3 scripts/datasets/bootstrap_camponotus_autolabel.py \
  --images-root datasets/camponotus_raw/in_situ \
  --backend yolo \
  --yolo-weights experiments/yolo/camponotus_yolo26n_v2/weights/best.pt \
  --conf 0.45 \
  --with-tracking \
  --tracker botsort \
  --botsort-with-reid \
  --state-priority-soft \
  --state-priority-iou-thresh 0.70 \
  --state-priority-score-gap-max 0.12 \
  --out datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_botsort_reid_soft.json \
  --cvat-video-xml-out datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_botsort_reid_soft_cvat_video.xml
```

This writes CVAT Video 1.1 XML with:
- tracked objects as `<track>` items,
- per-frame `<box>` entries,
- `state` attribute (`normal` / `trophallaxis`) on each box.

### Optional: compare tracked vs ordinary prelabels

Generate an ordinary (non-tracked) JSON first (same weights/conf, just without `--with-tracking`), then compare:

```bash
python3 scripts/evaluation/compare_camponotus_prelabels_tracking.py \
  --ordinary datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_plain.json \
  --tracked datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked.json \
  --out-json experiments/results/camponotus_prelabels_tracking_compare.json \
  --out-txt experiments/results/camponotus_prelabels_tracking_compare.txt
```

This writes a reproducible log with:
- annotation count deltas,
- per-image density changes,
- `track_id` coverage,
- track length distribution and short-track ratio,
- gap events (flicker proxy).

### One-class CVAT command (ant + state attribute)

For Pattern A tasks (single rectangle label `ant` with `state` attribute), convert tracked prelabels to one class:

```bash
python3 scripts/datasets/coco_shift_category_ids_for_cvat.py \
  --in datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_botsort_reid_soft.json \
  --out datasets/camponotus_processed/prelabels/camponotus_prelabels_yolo26n_v2_tracked_botsort_reid_soft_cvat_ant_only.json \
  --collapse-to-single-label ant \
  --carry-state-attributes
```

This keeps only `ant` as the CVAT label and writes behavior to annotation `attributes.state` (`normal` / `trophallaxis`).

### Optional: video tracking visualization (IDs + trajectory lines)

For qualitative QA on a full clip, render tracked boxes with `id` labels and history trails:

```bash
python3 scripts/inference/track_yolo_video.py \
  --weights experiments/yolo/camponotus_yolo26n_v2/weights/best.pt \
  --source-video /path/to/camponotus_trophallaxis_006.mov \
  --out-video runs/detect/experiments/results/video_try_troph0033/camponotus_trophallaxis_006_tracked_botsort.mp4 \
  --tracker botsort \
  --conf 0.45 \
  --trail-len 40
```

Notes:
- Use repo-relative output paths under `runs/` (or another project output folder).
- Keep source path configurable per machine (`/path/to/...`) and avoid hardcoding local absolute paths in committed scripts/docs.
- Switch `--tracker` to `bytetrack` for side-by-side qualitative comparison.

## 7) Label Studio Fallback

CVAT is the primary tool. Use Label Studio only when:

- CVAT deployment is unavailable,
- or team preference requires Label Studio for a specific annotation stage.

If Label Studio is used:

- export to COCO,
- ensure class names and ids map to the same schema,
- run conversion/validation scripts before dataset export.

## 8) Pre-Export QA

Before final export:

- run random visual checks on each batch,
- verify trophallaxis pair labeling policy is followed,
- ensure no unexpected classes are present,
- verify files are loadable and image paths resolve in downstream scripts.

After `prepare_camponotus_detection_dataset.py`, run `validate_camponotus_dataset.py` on the YOLO/COCO outputs.

## 9) Annotation hygiene (Ideas 1–3)

- **`state` → class 1** on export is handled by `prepare_camponotus_detection_dataset.py` (see `--state-attr` / `--trophallaxis-state-value`). Keep values consistent with CVAT label definitions.
- **`track_id`:** Optional in CVAT; exported on COCO annotations when present. [`evaluate.py`](../scripts/evaluation/evaluate.py) ignores `track_id` for mAP. For **sequence / Idea 2** work, keep the same physical ant on the same `track_id` within a clip when feasible. Use `validate_camponotus_dataset.py --strict-track-id` if you need hard guarantees.
- **RF-DETR layout:** After canonical prep, run `scripts/datasets/prepare_camponotus_coco_rfdetr.py` (or set `CAMPO_PREP_RFDETR_COCO=1` on `scripts/run_camponotus_dataset_workflow.sh`) to populate `datasets/camponotus_rfdetr_coco/`.

## 10) Research roadmap (Ideas 1–3)

Single-frame two-class detection (YOLO vs RF-DETR), then optional interaction and segmentation phases, without relabeling from scratch — see [`docs/camponotus_research_roadmap.md`](camponotus_research_roadmap.md).
