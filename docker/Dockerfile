# Training image: Python 3.10+ with CUDA optional.
# For GPU hosts, use an NVIDIA base and matching PyTorch wheels, e.g.:
#   FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# Then install torch from https://pytorch.org/get-started/locally/
FROM python:3.10-slim-bookworm

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: CPU training (override with GPU image + torch CUDA build for real runs)
CMD ["python", "scripts/train/train_yolo.py"]
