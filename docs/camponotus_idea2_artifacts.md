# Camponotus Idea 2 — artifact index

Single place to learn **what each JSON is for**, **who produces it**, and **how runs are laid out**. Protocol details stay in [`camponotus_idea2_event_protocol.md`](camponotus_idea2_event_protocol.md).

## Machine-specific paths in JSON

Scripts often record absolute paths inside outputs (`source_coco_annotations`, `gt_events`, `pred_events`, etc.). Paths under `/media/...` or other mounts are **local to the machine that ran the command**. Treat them as provenance, not as required locations for reproducibility. Use the same **relative repo paths** documented below when sharing or re-running.

## Dataset files (version-controlled benchmarks)

| Kind | Path | Role | Produced by |
|------|------|------|-------------|
| Manual event GT | [`datasets/camponotus_idea2_event_benchmark_v1.json`](../datasets/camponotus_idea2_event_benchmark_v1.json) | Canonical evaluation ground truth (clips + events). | Human QA |
| Auto-draft GT | [`datasets/camponotus_idea2_event_benchmark_v1_auto.json`](../datasets/camponotus_idea2_event_benchmark_v1_auto.json) | Heuristic draft from CVAT COCO `state` + geometry; **not** a substitute for `v1` without review. | [`scripts/datasets/build_camponotus_idea2_event_gt.py`](../scripts/datasets/build_camponotus_idea2_event_gt.py) |

## Experiment outputs (per run)

**Default layout for new one-shot runs:** `experiments/results/idea2/<RUN_NAME>/`

Set `RUN_NAME` (and optionally `OUT_DIR`) when calling [`scripts/run_camponotus_idea2_hybrid.sh`](../scripts/run_camponotus_idea2_hybrid.sh). Artifact filenames use the pattern `${RUN_NAME}_<suffix>.json`.

| Kind | Typical filename (under run dir) | Role | Produced by |
|------|----------------------------------|------|-------------|
| COCO prelabels | `<RUN_NAME>_prelabels_coco.json` | Full detection export per image (large). | [`scripts/datasets/bootstrap_camponotus_autolabel.py`](../scripts/datasets/bootstrap_camponotus_autolabel.py) (via hybrid script) |
| Prelabels manifest | `<RUN_NAME>_prelabels_coco_manifest.json` | Small summary of prelabel run. | bootstrap (if emitted) |
| MOT JSON | `<RUN_NAME>_mot.json` | Tracker rows (`frame`, `track_id`, `bbox`, …) for event inference. | bootstrap `--mot-out-json` |
| Predicted events (helper) | `<RUN_NAME>_events_with_helper.json` | Inferred trophallaxis event intervals. | [`scripts/inference/infer_camponotus_idea2_events.py`](../scripts/inference/infer_camponotus_idea2_events.py) |
| Predicted events (no helper) | `<RUN_NAME>_events_no_helper.json` | Same, `--disable-helper-signal`. | infer script |
| Evaluation | `<RUN_NAME>_events_{with,no}_helper_eval.json` | TP/FP/FN vs **canonical** `v1` GT. | [`scripts/evaluation/evaluate_camponotus_idea2_events.py`](../scripts/evaluation/evaluate_camponotus_idea2_events.py) |
| Compare bundle | `<RUN_NAME>_events_helper_vs_no_helper.json` | Delta between two eval JSONs. | [`scripts/evaluation/compare_camponotus_idea2_event_metrics.py`](../scripts/evaluation/compare_camponotus_idea2_event_metrics.py) |
| Tracker→CVAT map | `<RUN_NAME>_tracker_to_cvat_map.json` (optional `_iou015` etc.) | Per-sequence alignment tracker IDs → CVAT `track_id`. | [`scripts/evaluation/map_camponotus_tracker_ids.py`](../scripts/evaluation/map_camponotus_tracker_ids.py) |

### Naming suffix legend (ad-hoc analyses)

These suffixes appear when comparing **different ground-truth or ID assumptions**. Do not delete blindly; each answers a different question.

| Suffix | Meaning |
|--------|--------|
| `_eval` | Standard eval vs `datasets/camponotus_idea2_event_benchmark_v1.json`. |
| `_eval_auto_gt` | Eval vs **auto-draft** GT (`*_v1_auto.json`) — diagnostic only. |
| `_remapped` | Predicted `track_id_a/b` remapped through a tracker→CVAT map before eval. |
| `_remapped_iou015` (or similar) | Same remap built with a different `--min-iou` for the mapper. |
| `_helper_vs_no_helper` | Compare output from `compare_camponotus_idea2_event_metrics.py`. |

## Legacy / flat paths

Older commands wrote Idea 2 JSONs directly under `experiments/results/` (no `idea2/` subfolder), for example:

- `camponotus_idea2_events_hybrid_with_helper.json`
- `camponotus_idea2_events_hybrid_*_eval.json`

Those are valid artifacts; prefer **`experiments/results/idea2/<RUN_NAME>/`** for new runs so everything for one experiment stays in one directory.

## Regenerating large prelabels COCO

If `*_prelabels_coco.json` is gitignored or removed locally, regenerate by re-running the hybrid pipeline (or bootstrap alone) with the same `IN_SITU_ROOT`, `SEQ_LIST`, weights, and `OUT_DIR` / `RUN_NAME`:

```bash
GT_EVENTS=datasets/camponotus_idea2_event_benchmark_v1.json \
IN_SITU_ROOT="/path/to/in_situ" \
RUN_NAME=my_run \
OUT_DIR=experiments/results/idea2/my_run \
bash scripts/run_camponotus_idea2_hybrid.sh
```

The script writes `<RUN_NAME>_prelabels_coco.json` inside `OUT_DIR`.

## Related docs

- [`cli_commands.md`](cli_commands.md) — copy-paste commands for Idea 2.
- [`datasets.md`](datasets.md) — Camponotus dataset layout; pointer to this file for benchmarks.
- [`camponotus_research_roadmap.md`](camponotus_research_roadmap.md) — Idea 2 scope and tooling.
