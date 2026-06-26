#!/usr/bin/env python3
"""COCO detection dataset for torchvision Faster R-CNN (Camponotus category_id 0/1 → labels 1/2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def coco_xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def coco_category_to_label(category_id: int) -> int:
    """Map benchmark COCO category_id (0/1) to torchvision foreground label (1/2)."""
    return int(category_id) + 1


def label_to_coco_category(label: int) -> int:
    """Map torchvision label (1/2) back to benchmark category_id (0/1)."""
    return int(label) - 1


class CocoDetectionTorchvision(Dataset):
    """Load COCO instances JSON + image folder for Faster R-CNN training/eval."""

    def __init__(
        self,
        coco_json: Path,
        image_root: Path,
        *,
        train: bool = False,
        horizontal_flip_p: float = 0.5,
        max_images: int | None = None,
    ) -> None:
        self.image_root = Path(image_root).expanduser().resolve()
        self.train = train
        self.horizontal_flip_p = horizontal_flip_p
        data = json.loads(Path(coco_json).expanduser().resolve().read_text(encoding="utf-8"))

        images = sorted(data.get("images", []), key=lambda im: int(im["id"]))
        if max_images is not None:
            images = images[: max(0, max_images)]

        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            iid = int(ann["image_id"])
            anns_by_image.setdefault(iid, []).append(ann)

        self.samples: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for im in images:
            iid = int(im["id"])
            self.samples.append((im, anns_by_image.get(iid, [])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        im_meta, anns = self.samples[index]
        file_name = Path(str(im_meta["file_name"])).name
        path = self.image_root / file_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")

        pil = Image.open(path).convert("RGB")
        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            boxes.append(coco_xywh_to_xyxy(ann["bbox"]))
            labels.append(coco_category_to_label(int(ann["category_id"])))

        if boxes:
            box_t = torch.tensor(boxes, dtype=torch.float32)
            label_t = torch.tensor(labels, dtype=torch.int64)
        else:
            box_t = torch.zeros((0, 4), dtype=torch.float32)
            label_t = torch.zeros((0,), dtype=torch.int64)

        if self.train and self.horizontal_flip_p > 0:
            if torch.rand(1).item() < self.horizontal_flip_p:
                pil = TF.hflip(pil)
                width = pil.width
                if box_t.numel() > 0:
                    x1 = box_t[:, 0].clone()
                    x2 = box_t[:, 2].clone()
                    box_t[:, 0] = width - x2
                    box_t[:, 2] = width - x1

        image = TF.to_tensor(pil)
        target = {"boxes": box_t, "labels": label_t, "image_id": torch.tensor([int(im_meta["id"])])}
        return image, target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict]]:
    images, targets = zip(*batch)
    return list(images), list(targets)
