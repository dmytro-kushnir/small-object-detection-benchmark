# Camponotus Idea 2 Event Protocol (v1)

This protocol defines how Idea 2 converts tracked ants into trophallaxis events and how those events are scored.

## 1) Event unit

An event is an interval over a pair of identities in one sequence:

- `sequence_name`
- `track_id_a`
- `track_id_b` (unordered pair with `a < b`)
- `start_frame` (1-indexed, inclusive)
- `end_frame` (1-indexed, inclusive)

## 2) Frame-level pair eligibility

At frame `t`, pair `(a,b)` is marked active if the pair score exceeds threshold:

`score_t = w_iou * iou_t + w_dist * (1 - min(1, dist_t / max_dist_px)) + w_helper * helper_t`

Where:

- `iou_t`: IoU between boxes of tracks `a` and `b` in frame `t`
- `dist_t`: Euclidean distance between box centers
- `max_dist_px`: distance normalization scale (default 90 px)
- `helper_t`: optional Idea 1 helper signal
  - default: `1.0` if either detection is class `trophallaxis`, else `0.0`

Suggested defaults:

- `w_iou=0.55`, `w_dist=0.35`, `w_helper=0.10`
- `pair_score_threshold=0.45`

## 3) Temporal smoothing -> events

Given active/inactive per frame for pair `(a,b)`:

1. Build active runs.
2. Merge neighboring runs when gap length `<= max_gap_frames` (default `3`).
3. Keep merged runs with duration `>= min_active_frames` (default `12`).

Each retained run becomes one predicted event.

## 4) Ground-truth format (benchmark subset)

Ground truth event files must store:

- `version`
- `fps`
- `clips` list with:
  - `clip_id`
  - `sequence_name`
  - `events` list (same fields as event unit)

## 5) Event matching for evaluation

Predicted and GT events are matched with one-to-one greedy assignment under:

- same `sequence_name`
- same unordered pair `(track_id_a, track_id_b)`
- temporal IoU `>= match_tiou_threshold` (default `0.30`)

Temporal IoU:

`tIoU = overlap_frames / union_frames`

Metrics:

- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1`
- `mean_tiou_matched`

## 6) Diagnostics to report

- per-sequence TP/FP/FN
- pair-level confusion (which identity pairs are most often missed)
- track coverage diagnostics from MOT payload:
  - `num_tracks`, `short_track_ratio`, `gap_events`, `gap_frames_total`

## 7) Scope note

This protocol is for **Idea 2 event-level scoring**. It complements Idea 1 (frame-box class scoring), but does not replace Idea 1 detector benchmarks in `evaluate.py`.
