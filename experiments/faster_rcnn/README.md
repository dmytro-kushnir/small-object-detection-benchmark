# Faster R-CNN experiment artifacts

Training and evaluation outputs for torchvision Faster R-CNN baselines (Camponotus and future datasets).

## Layout per run

```
experiments/faster_rcnn/<run_id>/
  config.yaml
  metrics.json          # train/val loss per epoch
  system_info.json
  weights/best.pth
  weights/last.pth
  predictions_val.json
  predictions_test.json
  inference_benchmark_val.json
  inference_benchmark_test.json
```

Aggregated COCO metrics (from `evaluate.py`):

```
experiments/results/camponotus_faster_rcnn_trackidmajor_896_metrics_{val,test}.json
experiments/results/camponotus_faster_rcnn_vs_yolo8962_{val,test}.json
```

Weights and large JSONs are gitignored; use `scripts/run_camponotus_faster_rcnn_exp.sh` to reproduce.
