# EXP-A002b: Ant resolution sweep — recommendation

Generated from aggregated metrics (same `datasets/ants_yolo` val GT; YOLO26n, 20 epochs per resolution unless 640 reused from EXP-A000 full).

## Summary table (by imgsz)

| imgsz | mAP | mAP@0.5 | mAP_medium | P | R | FPS | latency ms |
|------:|----:|--------:|-----------:|--:|--:|----:|-----------:|
| 640 | 0.6358 | 0.9135 | 0.6361 | 0.9165 | 0.9363 | 58.1473 | 17.1977 |
| 768 | 0.6451 | 0.9222 | 0.6455 | 0.9139 | 0.9465 | 60.5712 | 16.5095 |
| 896 | 0.6264 | 0.9189 | 0.6277 | 0.9213 | 0.9429 | 60.0221 | 16.6605 |
| 1024 | 0.5237 | 0.9163 | 0.5289 | 0.9202 | 0.9431 | 57.6015 | 17.3607 |

## Best per metric

- **Highest mAP_medium:** imgsz **768** (0.6455).
- **Highest overall mAP (0.5:0.95):** imgsz **768** (0.6451).
- **Fastest FPS:** imgsz **768** (60.5712).
- **Lowest mean latency:** imgsz **768** (16.5095 ms).

## Trade-off recommendation

**Suggested imgsz (mAP_medium vs median FPS rule):** **768**.

*Rule used:* Among resolutions with FPS ≥ median FPS (59.0847), pick highest mAP_medium; tie-break lower imgsz.

