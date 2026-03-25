# Camponotus Labeling Guidelines (Detection)

This document defines the annotation policy for the Camponotus fellah detection dataset.

## Class Schema (exported training / evaluation)

After dataset preparation, every object is one of:

- `0`: `ant`
- `1`: `trophallaxis`

The same mapping is used in YOLO labels and in split COCO files under `datasets/camponotus_coco/annotations/`.

## CVAT authoring (recommended): one box label + `state` attribute

In CVAT you may use **a single rectangle label** (e.g. `ant`) for every ant instance and record behavior with an attribute:

- **`state`** (mutable): `normal` (or any value other than `trophallaxis`) vs `trophallaxis`
- When preparing the dataset, `scripts/datasets/prepare_camponotus_detection_dataset.py` maps:
  - `state == trophallaxis` → exported class `1`
  - any other `state` value → exported class `0`

Optional identity for future sequence / MOT-style work:

- **`track_id`** (number, typically **not** mutable): stable integer per ant within a clip (see `docs/camponotus_cvat_workflow.md`). Detection metrics do not use it; it is stored on exported COCO annotations when present.

## Legacy CVAT authoring: two box labels

You may instead draw boxes with two CVAT labels (`ant` and `trophallaxis`). If an annotation has **no** `state` attribute (or you disable state mapping with `--state-attr ""`), preparation uses the CVAT category only.

## Core Rule For Trophallaxis

If two ants are clearly engaged in trophallaxis in a frame, annotate **both ants** as trophallaxis for training:

- **Attribute workflow:** set **`state` = `trophallaxis`** on both boxes (boxes stay label `ant`).
- **Two-label workflow:** set both boxes to label `trophallaxis`.

Do not draw one box around the pair. Keep one box per ant.

## Labeling Rules

1. **Default (non-trophallaxis)**
   - All visible ants not clearly in trophallaxis: exported class `ant` (`0`) — e.g. `state = normal` or CVAT label `ant`.
2. **Trophallaxis**
   - If interaction is clearly visible and biologically consistent, both participating ants → exported class `trophallaxis` (`1`).
3. **Ambiguous frames**
   - If uncertain, use `ant` (`0`), not `trophallaxis`.
4. **Occlusion**
   - Label only if a meaningful visible body extent remains.
   - Skip extremely tiny/uncertain fragments.
5. **Bounding boxes**
   - Keep boxes tight around each ant body.
   - Avoid excessive background.
   - Maintain consistent style across annotators.
6. **Cut borders**
   - If ant is partially outside frame, annotate visible part only.
7. **Overlapping ants**
   - Keep separate boxes when separable.
   - If impossible to separate reliably, skip uncertain instance instead of noisy annotation.

## Consistency Checklist

Before export, check:

- Exported class logic: every box should become `0` or `1` as intended (via `state` or CVAT label).
- Trophallaxis pairs are both exported as class `1`.
- Ambiguous interactions are not over-labeled as `1`.
- Box tightness is consistent across sequences and sources.
- No duplicate boxes for the same ant instance in one frame.

## Examples (Decision Policy)

- Clear mouth-to-mouth exchange between two ants -> both `trophallaxis` (`1`).
- Two ants touching but behavior unclear -> both `ant` (`0`), unless evidence is clear.
- One ant clearly in trophallaxis, second partly visible but identifiable -> both `1` if confidence is high.
- Motion blur where interaction cannot be verified -> `0` for visible ants.

## Versioning

When rules change, update this file and record date + change summary in git history so dataset revisions remain reproducible.
