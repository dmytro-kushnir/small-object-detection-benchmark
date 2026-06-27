# EXP-CAMPO-FRCNN-TRACKIDMAJOR-896 — Faster R-CNN third baseline

**Status:** **Complete** (train + unified `evaluate.py` on 2511-image bundle, 2026-06-27).

**Question:** How does a classical **two-stage CNN** (Faster R-CNN R50-FPN) compare to YOLO26n and RF-DETR Small on the **2511-image track-majority** bundle under unified `evaluate.py`?

**What to run:**

```bash
./scripts/run_camponotus_faster_rcnn_exp.sh
# Re-infer / eval only:
EXP_FRCNN_SKIP_TRAIN=1 EXP_FRCNN_DEVICE=cuda:0 ./scripts/run_camponotus_faster_rcnn_exp.sh
```

Config: [`configs/camponotus_faster_rcnn_trackidmajor_896.yaml`](../../configs/camponotus_faster_rcnn_trackidmajor_896.yaml)

**Artifacts:**

| Artifact | Path |
|----------|------|
| Weights | `experiments/faster_rcnn/camponotus_faster_rcnn_trackidmajor_896/weights/best.pth` |
| Val/test metrics | `experiments/results/camponotus_faster_rcnn_trackidmajor_896_metrics_{val,test}.json` |
| vs YOLO8962 | `experiments/results/camponotus_faster_rcnn_vs_yolo8962_{val,test}.json` |

**Recorded results (RTX 4070, unified `evaluate.py`, `conf=0.25`):**

| Split | mAP@[.5:.95] | mAP@.50 | P | R | FPS |
|-------|-------------:|--------:|--:|--:|----:|
| Val | 0.411 | 0.825 | 0.657 | 0.866 | 9.6 |
| Test | 0.387 | 0.616 | 0.659 | 0.840 | 9.5 |

**vs YOLO26n `8962` (Δ = FRCNN − YOLO):** val mAP **+0.044**, mAP@.50 **+0.205**; test mAP **+0.023**, mAP@.50 **+0.069**. **vs RF-DETR `ep60_es`:** RF-DETR still higher on mAP@[.5:.95] (val **0.489** vs **0.411**).

**Paper role:** Extend **Table 5** / **Table 5a** with Faster R-CNN column (Reviewer 1 comment 3). See narrative in [`research_analysis.md`](../research_analysis.md) → **EXP-CAMPO-FRCNN-TRACKIDMAJOR-896**.

**Caveats (record in `evaluation_note`):**

- Torchvision **shortest-side** resize (`min_size=896`, `max_size=1333`), not YOLO square letterbox.
- **50-epoch** SGD schedule; not compute-matched to YOLO (~100 ep) or RF-DETR (60 ep).
- Best checkpoint by **val loss @ epoch 3** (later epochs overfit on val loss).
- Initial infer export (2026-06-26) was **invalid** (0 detections) due to symlink basename mismatch; fixed in `infer_faster_rcnn.py` + `iter_gt_aligned_image_paths` in `coco_pred_common.py`.

**Smoke (no Camponotus data):** `./scripts/run_faster_rcnn_smoke_test.sh`

**See also:** [`cli_commands.md`](../cli_commands.md), [`architecture.md`](../architecture.md).
