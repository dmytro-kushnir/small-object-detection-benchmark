# Camponotus research roadmap (detection vs interaction)

This document aligns the three research directions with **one canonical annotation** (CVAT → [`prepare_camponotus_detection_dataset.py`](../scripts/datasets/prepare_camponotus_detection_dataset.py)) and **derived exports** for later phases. See also [`docs/camponotus_cvat_workflow.md`](camponotus_cvat_workflow.md).

**Prior work (biology, citation for papers):** Greenwald, Segre & Feinerman (*Sci. Rep.* **5**, 12496, 2015) — dual-camera barcode tracking plus fluorescence imaging of crop loads in *Camponotus sanctus* / *C. fellah*; quantitative trophallactic networks; duration ≠ volume. DOI: <https://doi.org/10.1038/srep12496>. Synthesis and BibTeX: [`docs/research_analysis.md`](research_analysis.md) (Camponotus section, “Prior work”).

## Idea 1 — Two-class detection (current benchmark)

**Goal:** Treat trophallaxis as a second detection class on single frames (`ant` vs `trophallaxis`).

**Data:** `datasets/camponotus_yolo` / `datasets/camponotus_coco` (classes 0/1). Map CVAT `state` or two rectangle labels in the prepare script.

**Models:** YOLO26 ([`train_yolo.py`](../scripts/train/train_yolo.py)) and RF-DETR ([`train_rfdetr_ants.py`](../scripts/train/train_rfdetr_ants.py) with [`configs/expCAMPO_rfdetr.yaml`](../configs/expCAMPO_rfdetr.yaml) after [`prepare_camponotus_coco_rfdetr.py`](../scripts/datasets/prepare_camponotus_coco_rfdetr.py)).

**Evaluation:** Unified [`evaluate.py`](../scripts/evaluation/evaluate.py). **Inference** for RF-DETR on two classes uses [`infer_rfdetr.py`](../scripts/inference/infer_rfdetr.py) **`--class-id-mode multiclass`**.

**Orchestration:** [`scripts/run_camponotus_rfdetr_exp.sh`](../scripts/run_camponotus_rfdetr_exp.sh) (train → val/test infer → compare vs YOLO metrics).

### Idea 1 — Next steps (benchmark)

Synthesis table: [`research_analysis.md`](research_analysis.md) → **“Idea 1 — Paper-ready summary”**.

1. **RF-DETR @896 on the sequence-safe split** — Prepare Roboflow export from `datasets/camponotus_yolo/camponotus_full_export_unique_sequence_safe` into e.g. `datasets/camponotus_rfdetr_coco_sequence_safe`, train with `--resolution 896`, run `infer_rfdetr.py` → `bench_rfdetr.py` → `evaluate.py`, then [`compare_camponotus_rfdetr_vs_yolo.py`](../scripts/evaluation/compare_camponotus_rfdetr_vs_yolo.py) with baseline `camponotus_idea1_sequence_safe_full_896_metrics_{val,test}.json`. Fills **TBD** cells in the paper summary matrix and completes **RF-DETR640 vs RF-DETR896** on the **same** split (subject to schedule parity caveats).
2. **Within RF-DETR @640 vs @896** — On each split separately, optional `compare_metrics.py` between `camponotus_rfdetr_*_metrics_*.json` (640) and the new 896 metrics (same split label only).
3. **Optional latency row** — Re-benchmark RF-DETR with `EXP_A005_OPTIMIZE_INFERENCE=1` if you need optimized inference timing; re-verify mAP ([`infer_rfdetr.py`](../scripts/inference/infer_rfdetr.py)).
4. **Idea 2+** — After Idea 1 tables stabilize: event / MOT-style metrics for interaction modeling (see Idea 2 below); ant-only exports for detector training as needed.

### Idea 1 — Qualitative video (out-of-distribution)

**Benchmark vs new lab clips:** `evaluate.py` mAP on val/test measures **in-distribution** (same preparation pipeline as training). A **new** video (different lighting, camera, crop, or framing) is **domain shift**; tracking scripts (`track_yolo_video.py`, `track_rfdetr_video.py`) are for **inspection**, not a reported metric unless you add labels or a controlled evaluation.

**What analytics JSONs show today:** Aggregate `states` counts (per detection box over time), `unique_tracks`, short-track / gap stats, tracker settings. They support **global** comparisons (e.g. sequence-safe vs track_id–majority weights on the same clip) but **not** which identity was wrong until **per-track** summaries exist.

**Per-track state (debugging):** Re-run tracking with `--analytics-out`; the JSON now includes **`per_track`** — each track id’s `state_counts`, `dominant_state`, and `dominant_fraction` so you can see whether one ID stayed on `normal` while another was `trophallaxis`.

**Latest OOD discovery (trophallaxis-only clip):** On `trophalaxis_001_example.mp4` using YOLO `camponotus_idea1_trackidmajor_full_896` + soft state-priority (`--conf 0.25`, `--state-priority-soft`), analytics still report **normal-dominant** counts (`normal=698`, `trophallaxis=262`; 455 frames). `per_track` shows one long track largely correct (`id 1`: 211 troph / 159 normal) and one long track almost always normal (`id 2`: 442 normal / 1 troph), with only `relabel_count=9`. Treat this as label-policy/domain-shift failure evidence, not a tracker-only issue.

**When “everything should be trophallaxis” but the model shows `normal`:** That is **single-frame classifier error** on OOD data (or threshold too high), not something Idea 2 fixes by itself. Mitigations: **more / harder negatives and positives** in training, **domain-matched** clips, **lower `--conf`**, **`--state-priority-soft`** (YOLO/RF-DETR) where two boxes overlap, or **Idea 2** if you abandon two-class frame labels and treat interaction as **geometry + time** (see below).

**Idea 1 vs Idea 2 (practical):** Finish **Idea 1** benchmark rows you care about (e.g. sequence-safe RF-DETR @896) for **paper**; treat **OOD qualitative** as **failure analysis**, not a blocker. Move to **Idea 2** when you want **event-level** behavior and are ready to **train/evaluate** one-class detection + pairing rules or MOT metrics — **not** as a shortcut to fix OOD two-class confusion without new data or a different definition of “trophallaxis.”

### Idea 1 — Label audit checklist (before next retrain)

Use this checklist on a sample of clips with known interaction windows before spending more training cycles.

1. **Symmetry rule per frame:** if ant A and ant B are in trophallaxis, mark **both** boxes as `trophallaxis` in that frame (unless one ant is fully unobservable).
2. **Transition consistency:** define explicit start/end criteria (first and last positive frame) and apply the same tolerance for brief occlusions across annotators.
3. **Visibility exceptions:** when one ant is truncated/occluded, document allowed fallback (`normal` vs unlabeled) and keep it consistent across the dataset.
4. **Track-ID continuity QA:** for trophallaxis bouts, verify that `track_id` stays stable per ant; avoid identity swaps that create contradictory class history.
5. **Pair-level spot check:** for each audited bout, verify two long tracks have similar trophallaxis share; large asymmetry (e.g., one track >90% normal) is a red flag.
6. **Acceptance threshold for retrain:** in audited frames, require high pairwise agreement with the symmetry rule (target >=95%) before treating Idea 1 as a clean benchmark.

If checklist failures persist even after relabeling, prioritize Idea 2 event inference while keeping Idea 1 as a detection baseline.

## Idea 2 — Interaction modeling (tracking-based)

**Goal:** Detector outputs **one class** (`ant`); trophallaxis is inferred from **spatial–temporal** relations (e.g. proximity + minimum duration between track identities).

**Data (training):** Derived **ant-only** export — [`export_camponotus_ant_only_for_idea2.py`](../scripts/datasets/export_camponotus_ant_only_for_idea2.py) → `datasets/camponotus_coco_ant_only` / `datasets/camponotus_yolo_ant_only`. Each COCO annotation keeps a non-standard boolean **`trophallaxis_gt`** for future event-level evaluation (ignored by standard COCOeval). Preserve **`track_id`** in CVAT exports when practical.

**Data (evaluation, proposed):** Ground-truth **events** (pairs of `track_id`s or boxes with overlapping `trophallaxis_gt` in time) are **not** produced by this repo yet. After Idea 1 metrics, define explicitly:

- pairing rules (IoU / centroid distance thresholds),
- minimum contiguous frames for a positive event,
- how `state` / `trophallaxis_gt` on both ants maps to one interaction label.

**Metrics (future):** Event-centric precision/recall or temporal IoU — separate from bbox mAP.
Include MOT-style identity metrics (e.g., IDF1 / ID switches, optionally HOTA/MOTA) as a primary evaluation axis for Idea 2 tracking quality.

**Planning note (to avoid confusion):** MOT-style metrics are mainly an **Idea 2** requirement. For **Idea 1**, they can be used as optional diagnostics for tracker behavior, but model ranking should remain detection-first (COCO mAP + matched precision/recall).

**Relation to ants EXP-A006:** [`run_ants_expA006.sh`](../scripts/run_ants_expA006.sh) adds tracking + smoothing for **detection** stability, not behavioral interaction detection.

## Idea 3 — Segmentation-assisted interaction

**Status:** **Deferred** until Idea 2 is specified and it is clear that bbox proximity is insufficient.

**Scope:** Instance masks (manual, SAM-assisted, or trained segmenter) to refine contact/overlap; higher annotation or compute cost. No Camponotus segmentation pipeline in this repository yet.

## Annotation hygiene (all phases)

- Keep **`state`** consistent with the trophallaxis pair policy ([`docs/camponotus_labeling_guidelines.md`](camponotus_labeling_guidelines.md)).
- Use stable **`track_id`** per individual within a sequence when possible; run [`validate_camponotus_dataset.py`](../scripts/datasets/validate_camponotus_dataset.py) with **`--strict-track-id`** if you require at most one box per `(image_id, track_id)`.
- Prefer **one** CVAT export as source of truth; avoid maintaining parallel label trees.

## Dataset layout summary

| Artifact | Role |
|----------|------|
| `datasets/camponotus_yolo`, `datasets/camponotus_coco` | Idea 1 (canonical) |
| `datasets/camponotus_rfdetr_coco` | RF-DETR training layout (train/valid) |
| `datasets/camponotus_yolo_ant_only`, `datasets/camponotus_coco_ant_only` | Idea 2 detector train + `trophallaxis_gt` side signal |
