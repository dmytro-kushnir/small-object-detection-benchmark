# Experiment sheets (concluded runs)

This folder holds **one short sheet per experiment** we treat as **finished** for reporting: purpose, conclusion, and where metrics live. Step-by-step reproduction commands remain in [`experiments.md`](../experiments.md); interpretation and tables live in [`research_analysis.md`](../research_analysis.md).

## How to use these sheets

- **Writing the paper:** use the **Paper role** line on each sheet to map runs to manuscript tables (see also Table S2 in `docs/manuscript_camponotus_trophallaxis_case_study_draft.md`).
- **Re-running:** follow links to configs and shell scripts in [`experiments.md`](../experiments.md) and [`reproduction.md`](../reproduction.md).

## COCO128 harness (methodology / operating points)

| Sheet | ID | Paper role (typical) |
|-------|-----|----------------------|
| [EXP-000](EXP-000.md) | Smoke baseline | Pipeline sanity; not main results tables |
| [EXP-001](EXP-001.md) | Train-only small-box filter | Data-prep pilot; negative result on mAP_small |
| [EXP-002](EXP-002.md) | High `imgsz` (1280) | Resolution lever vs 320 smoke; mixed net effect at 1 epoch |
| [EXP-002b](EXP-002b.md) | Resolution sweep 640–1024 | Informs `imgsz` policy; 896 peak on this sweep |
| [EXP-003](EXP-003.md) | SAHI vs vanilla | Justifies full-frame default when 896 weights exist |

## Dense ants (MOT → COCO val)

| Sheet | ID | Paper role (typical) |
|-------|-----|----------------------|
| [EXP-A000](EXP-A000.md) | YOLO domain baseline | Canonical ant detector before architecture compare |
| [EXP-A002b](EXP-A002b.md) | Ant resolution sweep | **768** chosen for downstream YOLO vs RF-DETR on ants |
| [EXP-A003](EXP-A003.md) | SAHI @768 | Vanilla 768 preferred; supports no-SAHI policy |
| [EXP-A004](EXP-A004.md) | ANTS v1 dense refinement | Concluded; no rescue vs vanilla 768 (supporting) |
| [EXP-A005](EXP-A005.md) | RF-DETR vs YOLO | **Table 1** style architecture comparison (ants) |
| [EXP-A006](EXP-A006.md) | RF-DETR + ByteTrack + smoothing | **Table 3** temporal post-processing (ants) |

## *Camponotus* (trophallaxis case study)

| Sheet | ID | Paper role (typical) |
|-------|-----|----------------------|
| [EXP-CAMPO-FULL-2511-YOLO8962-RFDETR-EP60](EXP-CAMPO-FULL-2511-YOLO8962-RFDETR-EP60.md) | Track-majority 2511 export | **Tables 2 / 2a** headline YOLO vs RF-DETR |
| [EXP-CAMPO-FULL-2511-YOLO-S8962](EXP-CAMPO-FULL-2511-YOLO-S8962.md) | YOLO26s capacity | Text / supplement: small does not beat nano on headline mAP |
| [EXP-CAMPO-sequence-safe-comparison](EXP-CAMPO-sequence-safe-comparison.md) | Sequence-safe YOLO vs RF-DETR @896 | **Table 4** complementary generalization story |

Additional Camponotus runs (smokes, 1926-image frozen bundle, cross-split diagnostics) are documented inline in [`research_analysis.md`](../research_analysis.md#camponotus-exp-campo-001--yolo26n-ant--trophallaxis-cvat) rather than as separate sheets here.

## Related manuscript material

- Draft case study: [`../manuscript_camponotus_trophallaxis_case_study_draft.md`](../manuscript_camponotus_trophallaxis_case_study_draft.md)
- Roadmap (Ideas 1–3): [`../camponotus_research_roadmap.md`](../camponotus_research_roadmap.md)
