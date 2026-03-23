# Camponotus Labeling Guidelines (Detection)

This document defines the annotation policy for the Camponotus fellah detection dataset.

## Class Schema

- `0`: `ant`
- `1`: `trophallaxis`

The same mapping must be used in all exports (CVAT COCO, YOLO, final COCO).

## Core Rule For Trophallaxis

If two ants are clearly engaged in trophallaxis in a frame, annotate **both ants** as class `trophallaxis` (`1`).

Do not draw one box around the pair. Keep one box per ant.

## Labeling Rules

1. **Default class**
   - All visible ants not clearly in trophallaxis: class `ant` (`0`).
2. **Trophallaxis**
   - If interaction is clearly visible and biologically consistent, label both participating ants as class `trophallaxis` (`1`).
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

- Class IDs exactly match `0/1`.
- Trophallaxis pairs are both labeled as class `1`.
- Ambiguous interactions are not over-labeled as class `1`.
- Box tightness is consistent across sequences and sources.
- No duplicate boxes for the same ant instance in one frame.

## Examples (Decision Policy)

- Clear mouth-to-mouth exchange between two ants -> both `trophallaxis` (`1`).
- Two ants touching but behavior unclear -> both `ant` (`0`), unless evidence is clear.
- One ant clearly in trophallaxis, second partly visible but identifiable -> both `1` if confidence is high.
- Motion blur where interaction cannot be verified -> `0` for visible ants.

## Versioning

When rules change, update this file and record date + change summary in git history so dataset revisions remain reproducible.
