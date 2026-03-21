# Datasets

The repo includes `datasets/raw/` and `datasets/processed/` (placeholders tracked; actual data is gitignored). Downloads and prepared runs are written there from the repository root.

Use **normal directories** inside the clone for `datasets/`, `experiments/`, and `models/` (not symlinks to other mounts), so Git can track `.gitkeep` files and paths match the scripts.

## Preparation

From the repository root:

```bash
python scripts/datasets/prepare_dataset.py
```

Override paths and options with Hydra, for example:

```bash
python scripts/datasets/prepare_dataset.py \
  input.coco_json=/path/to/instances.json \
  input.images_dir=/path/to/images \
  output_dir=datasets/processed/my_run
```

## Output layout

Under `output_dir`:

- `annotations/instances_{train,val,test}.json` — COCO detection JSON
- `images/{train,val,test}/` — copied or resized images
- `labels/{train,val,test}/` — YOLO `.txt` labels (normalized)
- `dataset.yaml` — Ultralytics data config
- `prepare_manifest.json` — counts, filter params, seed

See `configs/prepare_dataset.yaml` for all options.
