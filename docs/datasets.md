# Datasets

The repo includes `datasets/raw/` and `datasets/processed/` (placeholders tracked; actual data is gitignored). Downloads and prepared runs are written there from the repository root.

Use **normal directories** inside the clone for `datasets/`, `experiments/`, and `models/` (not symlinks to other mounts), so Git can track `.gitkeep` files and paths match the scripts.

Some committed JSON uses **paths relative to the repository root** (for anything inside the clone). Paths that point at video or bulk data **outside** the repo may appear as ``<DATASETS_ROOT>/...``; treat that as “under your local datasets root” (for example the parent folder that contains `ants_videos/`). Scripts that emit new metrics use `scripts/repo_paths.py` for the same convention.

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

See `configs/prepare_dataset.yaml` for all options. Bbox filters use `filter.apply_to`: `all` (every split) or `train` (val/test GT unchanged—used for EXP-001 vs baseline).

## Camponotus Workflow (custom dataset)

For the two-class Camponotus dataset (`ant`, `trophallaxis`), use:

- Raw layout: `datasets/camponotus_raw/`
- Processed manifests: `datasets/camponotus_processed/`
- Exports: `datasets/camponotus_yolo/` and `datasets/camponotus_coco/`
- RF-DETR train layout (train/valid + `_annotations.coco.json`): `datasets/camponotus_rfdetr_coco/` via `scripts/datasets/prepare_camponotus_coco_rfdetr.py` (optional: `CAMPO_PREP_RFDETR_COCO=1` with `run_camponotus_dataset_workflow.sh`)
- Idea 2 ant-only derivative: `scripts/datasets/export_camponotus_ant_only_for_idea2.py` → `datasets/camponotus_yolo_ant_only/`, `datasets/camponotus_coco_ant_only/` (includes `trophallaxis_gt` on each annotation)

Main entrypoint:

```bash
./scripts/run_camponotus_dataset_workflow.sh
```

Supporting docs:

- `docs/camponotus_labeling_guidelines.md`
- `docs/camponotus_cvat_workflow.md`
- `docs/camponotus_research_roadmap.md`
