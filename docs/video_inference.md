# Video inference quick examples

Quick copy-paste commands for rendering model predictions on video.

## YOLO (tracking + IDs + trails)

Use this when you want tracker IDs and trajectories from a YOLO checkpoint.

```bash
python3 scripts/inference/track_yolo_video.py \
  --weights experiments/yolo/camponotus_idea1_trackidmajor_full_40ep_b8w4/weights/best.pt \
  --source-video "/media/dmytro/data/datasets/ants_videos/camponotus_trophallaxis_006.mov" \
  --out-video experiments/visualizations/camponotus_idea1_trackidmajor_full_40ep_b8w4_tracked_camponotus_006.mp4 \
  --imgsz 640 \
  --device 0 \
  --tracker botsort \
  --conf 0.35 \
  --analytics-out experiments/visualizations/camponotus_idea1_trackidmajor_full_40ep_b8w4_tracked_camponotus_006_analytics.json
```

Useful toggles:

- `--color-mode state` or `--color-mode id`
- `--state-priority-soft --state-priority-iou-thresh 0.7 --state-priority-score-gap-max 0.12`
- `--track-thresh`, `--match-thresh`, `--track-buffer`

## RF-DETR (tracking + IDs + trails via ByteTrack)

Use this when you want the same style of tracked video from an RF-DETR checkpoint.

```bash
python3 scripts/inference/track_rfdetr_video.py \
  --weights experiments/rfdetr/camponotus_rfdetr_trackidmajor/weights/best.pth \
  --source-video "/media/dmytro/data/datasets/ants_videos/camponotus_trophallaxis_006.mov" \
  --out-video experiments/visualizations/camponotus_rfdetr_trackidmajor_tracked_camponotus_006.mp4 \
  --model-class RFDETRSmall \
  --conf 0.35 \
  --track-thresh 0.25 \
  --match-thresh 0.8 \
  --track-buffer 30 \
  --trail-len 30 \
  --color-mode state \
  --optimize-for-inference \
  --analytics-out experiments/visualizations/camponotus_rfdetr_trackidmajor_tracked_camponotus_006_analytics.json
```

Notes:

- RF-DETR script currently uses ByteTrack backend for ID assignment.
- `--optimize-for-inference` can improve throughput on some setups.
- If your checkpoint class differs, replace `--model-class RFDETRSmall` accordingly.
