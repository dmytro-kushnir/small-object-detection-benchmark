# Results

Training runs write under `experiments/yolo/<run_id>/`:

- Ultralytics artifacts (`weights/`, `results.csv`, etc.)
- `config.yaml` — resolved training config
- `metrics.json` — summarized metrics when available
- `system_info.json` — Python, CUDA, GPU name

Aggregated summaries can be copied to `experiments/results/` for benchmarking.

For paper-oriented synthesis and per-experiment interpretation (updated over time), see [`research_analysis.md`](research_analysis.md).
