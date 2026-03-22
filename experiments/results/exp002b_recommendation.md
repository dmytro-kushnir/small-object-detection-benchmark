# EXP-002b: Resolution sweep — recommendation

Generated from aggregated metrics (same val GT as EXP-000; YOLO26n, 1 epoch per resolution).

## Summary table (by imgsz)

| imgsz | mAP | mAP@0.5 | mAP_small | mAP_medium | mAP_large | P | R | FPS | latency ms |
|------:|----:|--------:|----------:|-----------:|----------:|--:|--:|----:|-----------:|
| 640 | 0.4874 | 0.5992 | 0.1465 | 0.3502 | 0.7718 | 0.8261 | 0.5588 | 27.5052 | 36.3568 |
| 768 | 0.4849 | 0.5910 | 0.1712 | 0.3784 | 0.7260 | 0.7973 | 0.5784 | 23.2100 | 43.0848 |
| 896 | 0.4968 | 0.6403 | 0.2102 | 0.4230 | 0.6923 | 0.7470 | 0.6078 | 21.3011 | 46.9459 |
| 1024 | 0.4504 | 0.6172 | 0.1914 | 0.4315 | 0.5836 | 0.6705 | 0.5784 | 21.5434 | 46.4180 |

## Best per metric

- **Highest mAP_small:** imgsz **896** (0.2102).
- **Highest overall mAP (0.5:0.95):** imgsz **896** (0.4968).
- **Highest mAP_large:** imgsz **640** (0.7718).
- **Fastest FPS:** imgsz **640** (27.5052).
- **Lowest mean latency:** imgsz **640** (36.36 ms).

## Trade-off recommendation

**Suggested imgsz for small-object quality vs speed:** **768**.

*Rule used:* Among resolutions with FPS ≥ median FPS (22.3767), pick highest mAP_small; tie-break lower imgsz.

### How to read this

- Higher **imgsz** often raises **mAP_small** (finer input) but can hurt **overall mAP** or **mAP_large** after short training, and usually lowers **FPS**.
- Use the table above to see whether a mid resolution improves **mAP_small** without giving up as much **speed** as the largest size.

