#!/usr/bin/env bash
# Minimal Faster R-CNN pipeline smoke test (synthetic 4-image COCO subset, 1 epoch).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SMOKE_ROOT="$ROOT/experiments/smoke_faster_rcnn_coco"
IMG_TRAIN="$SMOKE_ROOT/images/train"
IMG_VAL="$SMOKE_ROOT/images/val"
ANN="$SMOKE_ROOT/annotations"
OUT_DIR="$ROOT/experiments/faster_rcnn/smoke_camponotus_frcnn"

rm -rf "$SMOKE_ROOT" "$OUT_DIR"
mkdir -p "$IMG_TRAIN" "$IMG_VAL" "$ANN" "$OUT_DIR/weights"

python3 <<'PY'
import json
from pathlib import Path

from PIL import Image, ImageDraw

root = Path("experiments/smoke_faster_rcnn_coco")
for split, n_img, start_id in [("train", 3, 1), ("val", 1, 4)]:
    images = []
    anns = []
    aid = 1
    for i in range(n_img):
        iid = start_id + i
        name = f"img_{iid:04d}.jpg"
        path = root / "images" / split / name
        img = Image.new("RGB", (320, 240), color=(40, 40, 40))
        d = ImageDraw.Draw(img)
        d.rectangle((80, 60, 140, 120), outline=(0, 255, 0))
        d.rectangle((160, 100, 220, 160), outline=(255, 0, 0))
        img.save(path, quality=90)
        images.append({"id": iid, "file_name": name, "width": 320, "height": 240})
        anns.append({"id": aid, "image_id": iid, "category_id": 0, "bbox": [80, 60, 60, 60], "area": 3600, "iscrowd": 0})
        aid += 1
        anns.append({"id": aid, "image_id": iid, "category_id": 1, "bbox": [160, 100, 60, 60], "area": 3600, "iscrowd": 0})
        aid += 1
    coco = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": 0, "name": "normal"}, {"id": 1, "name": "trophallaxis"}],
    }
    (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
print("Wrote synthetic smoke COCO under experiments/smoke_faster_rcnn_coco")
PY

cat > /tmp/smoke_frcnn.yaml <<EOF
coco_train: experiments/smoke_faster_rcnn_coco/annotations/instances_train.json
coco_val: experiments/smoke_faster_rcnn_coco/annotations/instances_val.json
images_train: experiments/smoke_faster_rcnn_coco/images/train
images_val: experiments/smoke_faster_rcnn_coco/images/val
output_dir: experiments/faster_rcnn/smoke_camponotus_frcnn
min_size: 320
max_size: 512
epochs: 1
batch_size: 1
grad_accum_steps: 1
lr: 0.005
num_workers: 0
conf_threshold: 0.25
seed: 42
device: cpu
EOF

python3 scripts/train/train_faster_rcnn.py --config /tmp/smoke_frcnn.yaml

WEIGHTS="$OUT_DIR/weights/best.pth"
GT_VAL="$ANN/instances_val.json"

python3 scripts/inference/infer_faster_rcnn.py \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$OUT_DIR/predictions_val.json" \
  --conf 0.25 \
  --min-size 320 \
  --max-size 512 \
  --device cpu

python3 scripts/evaluation/bench_faster_rcnn.py \
  --weights "$WEIGHTS" \
  --source "$IMG_VAL" \
  --coco-gt "$GT_VAL" \
  --out "$OUT_DIR/inference_benchmark_val.json" \
  --min-size 320 \
  --max-size 512 \
  --device cpu

mkdir -p experiments/results
python3 scripts/evaluation/evaluate.py \
  --gt "$GT_VAL" \
  --pred "$OUT_DIR/predictions_val.json" \
  --weights "$WEIGHTS" \
  --images-dir "$IMG_VAL" \
  --out experiments/results/smoke_faster_rcnn_metrics_val.json \
  --experiment-id EXP-SMOKE-FRCNN \
  --train-config /tmp/smoke_frcnn.yaml \
  --inference-benchmark-json "$OUT_DIR/inference_benchmark_val.json" \
  --device cpu

echo "Faster R-CNN smoke test OK."
