# EXP-003: SAHI sliced inference — summary

Generated from `compare_metrics.py` outputs. **Compare − baseline** in tables below.

## vs vanilla EXP-000 (`test_run_metrics.json`)

- Baseline experiment: `EXP-000`
- SAHI (test_run weights): `EXP-003-sahi-base`

| Metric | Δ (SAHI − vanilla) |
|--------|---------------------|
| mAP@[.5:.95] | +0.035044 |
| mAP@0.5 | +0.068688 |
| mAP_small | +0.000000 |
| mAP_medium | +0.098089 |
| mAP_large | -0.001505 |
| Precision (matched) | -0.107143 |
| Recall (matched) | +0.068627 |
| FPS | +8.453264 |
| Latency mean (ms) | -10.967321 |

*Note:* SAHI sliced inference vs vanilla EXP-000 (test_run); same val GT; training unchanged. Vanilla benchmark used plain YOLO predict; SAHI row uses sliced FPS from evaluate.py --sahi-config.

## vs vanilla high-res 896 (`exp002b_imgsz896_metrics.json`)

- Baseline experiment: `EXP-002b-imgsz896`
- SAHI (896-trained weights): `EXP-003-sahi-896`

| Metric | Δ (SAHI − vanilla 896) |
|--------|-------------------------|
| mAP@[.5:.95] | -0.036050 |
| mAP@0.5 | -0.000938 |
| mAP_small | -0.031467 |
| mAP_medium | -0.037376 |
| mAP_large | -0.055710 |
| Precision (matched) | -0.060241 |
| Recall (matched) | -0.049020 |
| FPS | +0.719986 |
| Latency mean (ms) | -1.534906 |

*Note:* SAHI sliced inference with weights trained at imgsz=896 (EXP-002b) vs vanilla 896 predict; same val GT.

## Interpretation (template)

- **mAP_small:** Positive deltas suggest SAHI helps small instances on this val set; compare magnitude to the gain from training at higher `imgsz` (EXP-002b vanilla 896 vs test_run vanilla) using your stored metrics JSONs.
- **Overall mAP / mAP_large:** Slicing can add false positives or duplicate boxes; watch precision and large-object AP.
- **Speed:** SAHI usually runs **more** forward passes per image than plain `predict`; expect lower FPS vs vanilla at the same weights unless slices are very large.
- **Vs raising resolution:** High `imgsz` changes the whole-image resize; SAHI keeps full resolution but adds overlap and multiple windows — benefits and costs differ; use FPS and mAP_small together to judge trade-offs for ANTS-style design.

**Sources:** [`exp003_sahi_vs_baseline.json`](exp003_sahi_vs_baseline.json), [`exp003_sahi_vs_exp002b_896.json`](exp003_sahi_vs_exp002b_896.json).
