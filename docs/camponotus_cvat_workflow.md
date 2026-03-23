# Camponotus CVAT Workflow

This workflow describes practical annotation for Camponotus fellah detection with two classes:

- `ant` (class id `0`)
- `trophallaxis` (class id `1`)

Canonical intermediate export for this project is **CVAT COCO**.

## 1) Prepare Input Data

Expected raw layout:

- `datasets/camponotus_raw/in_situ/seq_*/`
- `datasets/camponotus_raw/external/images/`

For video-derived data, extract frames first using:

- `scripts/datasets/extract_camponotus_frames.py`

## 2) Create CVAT Task(s)

Recommended setup:

1. Create one task per logical batch (for example by source or sequence groups).
2. Upload images as image lists (preserving sequence ordering for `in_situ`).
3. Define labels:
   - `ant`
   - `trophallaxis`

Tip: Keep `in_situ` and `external` in separate tasks or projects for easier auditing.

## 3) Annotate Bounding Boxes

Follow:

- `docs/camponotus_labeling_guidelines.md`

Critical rule:

- if two ants are in trophallaxis, both ants are labeled `trophallaxis`.

## 4) Interpolation / Tracking Usage

For sequential in-situ frames:

- Use CVAT tracking/interpolation tools to speed up repeated boxes.
- Always manually review keyframes and transitions.
- Re-check class assignments near behavioral transitions (ant <-> trophallaxis).

For external still images:

- Use standard box annotation (no interpolation needed).

## 5) Export

Export annotations as **COCO 1.0** from CVAT and store under:

- `datasets/camponotus_processed/annotations/cvat_coco/`

Recommended naming:

- `cvat_in_situ_batch01.json`
- `cvat_external_batch01.json`
- or merged `cvat_camponotus_all.json`

Keep original exports immutable. Any corrections should produce a new versioned file.

## 6) Prelabels (Semi-Automatic Workflow)

If using model-assisted prelabeling:

1. Generate prelabels with:
   - `scripts/datasets/bootstrap_camponotus_autolabel.py`
2. Import prelabels into CVAT.
3. Correct manually.
4. Export corrected COCO and keep both files:
   - machine prelabels (raw)
   - corrected human-reviewed labels

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
