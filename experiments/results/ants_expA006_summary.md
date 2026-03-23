# EXP-A006: RF-DETR + ByteTrack + temporal smoothing (ants val)

## 1. Method

- Detector: RF-DETR optimized baseline from EXP-A005.
- Tracking: `supervision.ByteTrack` (sequence-aware resets).
- Smoothing: drop short tracks (<3), fill 1-frame gaps, track-average score.

## 2. Metrics (absolute)

| Metric | A005 opt baseline | A006 tracking |
|--------|------------------:|--------------:|
| mAP@[.5:.95] | 0.6634 | 0.6635 |
| mAP@.5 | 0.9309 | 0.9357 |
| mAP_medium | 0.6639 | 0.6645 |
| Matched P | 0.9227 | 0.9388 |
| Matched R | 0.9615 | 0.9553 |

## 3. Delta (A006 - baseline)

- mAP@[.5:.95]: 0.000144
- mAP@.5: 0.004782
- mAP_medium: 0.000533
- Precision: 0.016135
- Recall: -0.006262
- FPS: -2.8282
- Latency mean (ms): 2.7797

## 4. Observations

- Review track overlays and before/after panels for false-positive removal and stability.
- Validate whether recall gains/losses align with filled-gap behavior.

## 5. Conclusion

- Decide whether temporal modeling improves detector-only baseline for this dataset/runtime.
