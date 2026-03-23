# `datasets/ants_coco/` (generated)

Roboflow **RF-DETR** COCO layout for the **same** ants train/val split as `datasets/ants_yolo/`.

**Do not commit** bulk data (directory is gitignored). Regenerate:

```bash
python3 scripts/datasets/prepare_ants_coco_rfdetr.py \
  --config configs/datasets/ants_coco_rfdetr.yaml
```

## Layout

| Path | Purpose |
|------|---------|
| `train/_annotations.coco.json` | COCO train (required by `rfdetr` autodetect) |
| `train/*.jpg` | Training images (basenames match `file_name` in JSON) |
| `valid/_annotations.coco.json` | COCO val (**`valid/`** is RF-DETR convention; same files as YOLO `images/val/`) |
| `valid/*.jpg` | Validation images |
| `annotations/instances_{train,val}.json` | Mirrors of the same COCO dicts (for tools expecting `annotations/`) |
| `ants_coco_manifest.json` | Provenance (counts, `git_rev`, optional hash of `ants_yolo/prepare_manifest.json`) |

Category id remains **0** (`ant`) to match `ants_yolo` COCO and YOLO predictions.
