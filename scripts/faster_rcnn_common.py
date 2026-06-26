#!/usr/bin/env python3
"""Shared Faster R-CNN (torchvision) helpers for train / infer / bench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def coco_category_to_label(category_id: int) -> int:
    """Map benchmark COCO category_id (0/1) to torchvision foreground label (1/2)."""
    return int(category_id) + 1


def label_to_coco_category(label: int) -> int:
    """Map torchvision label (1/2) back to benchmark category_id (0/1)."""
    return int(label) - 1


def build_faster_rcnn(
    *,
    num_classes: int = 3,
    min_size: int = 896,
    max_size: int = 1333,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Build Faster R-CNN R50-FPN with replaced box head for custom classes.

    ``num_classes`` is torchvision convention: background + foreground (e.g. 3 for
    two task classes mapped to labels 1 and 2).
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    if pretrained:
        model = fasterrcnn_resnet50_fpn(weights=weights)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
    model.transform.min_size = (int(min_size),)
    model.transform.max_size = int(max_size)
    return model


def load_faster_rcnn_checkpoint(
    weights_path: Path,
    *,
    num_classes: int = 3,
    min_size: int = 896,
    max_size: int = 1333,
    device: torch.device,
    pretrained_backbone: bool = False,
) -> torch.nn.Module:
    model = build_faster_rcnn(
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        pretrained=pretrained_backbone,
    )
    state = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def xyxy_to_xywh(boxes: torch.Tensor) -> list[list[float]]:
    out: list[list[float]] = []
    for x1, y1, x2, y2 in boxes.tolist():
        out.append([float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))])
    return out


def default_device_str() -> str:
    """Best available backend: CUDA → Apple MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device_str: str | None) -> torch.device:
    if device_str is None or device_str.strip().lower() == "auto":
        return torch.device(default_device_str())
    dev = device_str.strip()
    if dev.isdigit():
        dev = f"cuda:{dev}"
    return torch.device(dev)
