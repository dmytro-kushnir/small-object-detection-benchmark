#!/usr/bin/env python3
"""Faster R-CNN inference → COCO list JSON (evaluate.py compatible)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as TF

_INF = Path(__file__).resolve().parent
_SCRIPTS = _INF.parent
for _p in (_INF, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coco_pred_common import (  # noqa: E402
    iter_gt_aligned_image_paths,
    write_coco_predictions_json,
)
from faster_rcnn_common import (  # noqa: E402
    label_to_coco_category,
    load_faster_rcnn_checkpoint,
    resolve_device,
    xyxy_to_xywh,
)
def _predict_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    conf: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(Image.fromarray(rgb)).to(device)
    with torch.no_grad():
        out = model([tensor])[0]
    scores = out["scores"]
    keep = scores >= conf
    return out["boxes"][keep], out["labels"][keep], scores[keep]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True, help="Image file or directory")
    p.add_argument("--coco-gt", type=str, required=True, help="COCO GT for file_name → image_id")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--min-size", type=int, default=896)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    gt_path = Path(args.coco_gt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    device = resolve_device(args.device)

    model = load_faster_rcnn_checkpoint(
        weights,
        min_size=args.min_size,
        max_size=args.max_size,
        device=device,
    )

    work = iter_gt_aligned_image_paths(source, gt_path)
    if args.max_images is not None:
        work = work[: max(0, args.max_images)]

    detections: list[dict] = []
    for ip, image_id in work:
        try:
            boxes, labels, scores = _predict_image(model, ip, device, float(args.conf))
        except RuntimeError as exc:
            print(f"Skip {ip}: {exc}", file=sys.stderr)
            continue
        for box, label, score in zip(boxes.cpu(), labels.cpu(), scores.cpu()):
            lid = int(label.item())
            if lid <= 0:
                continue
            cat = label_to_coco_category(lid)
            xywh = xyxy_to_xywh(box.unsqueeze(0))[0]
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cat),
                    "bbox": xywh,
                    "score": float(score.item()),
                }
            )

    write_coco_predictions_json(out_path, detections)
    print(f"Wrote {len(detections)} detections → {out_path}")


if __name__ == "__main__":
    main()
