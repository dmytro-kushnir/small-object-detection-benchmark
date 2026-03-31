# Methodology

Experiments are driven by Hydra configs under `configs/`. Fixed random seeds, split ratios, and small-object filter thresholds are defined in `configs/prepare_dataset.yaml` and training hyperparameters in `configs/train/yolo.yaml`.

See [datasets.md](datasets.md) for the preparation pipeline and [results.md](results.md) for where metrics are stored.

For **copy-paste CLI** used in Camponotus and evaluation (train/infer/`evaluate.py`/`compare_*`), see [cli_commands.md](cli_commands.md).
