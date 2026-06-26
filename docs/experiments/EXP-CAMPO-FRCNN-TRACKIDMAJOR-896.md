# EXP-CAMPO-FRCNN-TRACKIDMAJOR-896 — Faster R-CNN third baseline

**Status:** Pipeline implemented; **full Camponotus metrics pending** GPU train on local dataset.

**Question:** How does a classical **two-stage CNN** (Faster R-CNN R50-FPN) compare to YOLO26n and RF-DETR Small on the **2511-image track-majority** bundle under unified `evaluate.py`?

**What to run:**

```bash
./scripts/run_camponotus_faster_rcnn_exp.sh
```

Config: [`configs/camponotus_faster_rcnn_trackidmajor_896.yaml`](../../configs/camponotus_faster_rcnn_trackidmajor_896.yaml)

**Expected artifacts:**

| Artifact | Path |
|----------|------|
| Weights | `experiments/faster_rcnn/camponotus_faster_rcnn_trackidmajor_896/weights/best.pth` |
| Val/test metrics | `experiments/results/camponotus_faster_rcnn_trackidmajor_896_metrics_{val,test}.json` |
| vs YOLO8962 | `experiments/results/camponotus_faster_rcnn_vs_yolo8962_{val,test}.json` |

**Paper role:** Extend **Table 5** / **Table 5a** with Faster R-CNN column (Reviewer 1 comment 3).

**Caveats (record in `evaluation_note`):**

- Torchvision **shortest-side** resize (`min_size=896`, `max_size=1333`), not YOLO square letterbox.
- **50-epoch** SGD schedule in default config; not compute-matched to YOLO (100 ep) or RF-DETR (60 ep).

**Smoke (no Camponotus data):** `./scripts/run_faster_rcnn_smoke_test.sh`

**See also:** [`cli_commands.md`](../cli_commands.md), [`architecture.md`](../architecture.md).
