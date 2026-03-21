# Datasets

If `datasets/` is not present, create it as a directory or **symlink** it to your large-data mount (see `AGENTS.md`).

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
