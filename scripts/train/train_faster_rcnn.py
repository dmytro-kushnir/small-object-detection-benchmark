#!/usr/bin/env python3
"""Fine-tune torchvision Faster R-CNN on Camponotus COCO exports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parents[1]
_DATASETS = _SCRIPTS / "datasets"
for _p in (_SCRIPTS, _DATASETS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coco_detection_torchvision import CocoDetectionTorchvision, collate_fn  # noqa: E402
from faster_rcnn_common import build_faster_rcnn, resolve_device  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(p: Path, root: Path) -> Path:
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def _git_rev(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _system_info() -> dict[str, Any]:
    import platform

    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_0"] = torch.cuda.get_device_name(0)
    return info


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int,
) -> float:
    model.train()
    running = 0.0
    n_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values()) / grad_accum_steps
        loss.backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        running += float(sum(v.item() for v in loss_dict.values()))
        n_steps += 1
    return running / max(n_steps, 1)


@torch.no_grad()
def _eval_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    n_steps = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        running += float(sum(v.item() for v in loss_dict.values()))
        n_steps += 1
    return running / max(n_steps, 1)


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=None, help="YAML config path.")
    p.add_argument("--coco-train", type=str, default=None)
    p.add_argument("--coco-val", type=str, default=None)
    p.add_argument("--images-train", type=str, default=None)
    p.add_argument("--images-val", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-size", type=int, default=None)
    p.add_argument("--max-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-train-images", type=int, default=None)
    p.add_argument("--max-val-images", type=int, default=None)
    args = p.parse_args()

    cfg: dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config).expanduser().resolve()
        if not cfg_path.is_file():
            print(f"Config not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg = dict(raw) if isinstance(raw, dict) else {}

    def _set(key: str, cli_val: Any, cfg_key: str | None = None) -> None:
        ck = cfg_key or key
        if cli_val is not None:
            cfg[ck] = cli_val

    _set("coco_train", args.coco_train)
    _set("coco_val", args.coco_val)
    _set("images_train", args.images_train)
    _set("images_val", args.images_val)
    _set("output_dir", args.output_dir)
    _set("epochs", args.epochs)
    _set("batch_size", args.batch_size)
    _set("grad_accum_steps", args.grad_accum_steps)
    _set("lr", args.lr)
    _set("min_size", args.min_size)
    _set("max_size", args.max_size)
    _set("num_workers", args.num_workers)
    _set("seed", args.seed)
    _set("device", args.device)
    _set("max_train_images", args.max_train_images)
    _set("max_val_images", args.max_val_images)

    required = ("coco_train", "coco_val", "images_train", "images_val", "output_dir")
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        print(f"Missing required settings: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    epochs = int(cfg.get("epochs", 50))
    batch_size = int(cfg.get("batch_size", 2))
    grad_accum = int(cfg.get("grad_accum_steps", 1))
    lr = float(cfg.get("lr", 0.005))
    min_size = int(cfg.get("min_size", 896))
    max_size = int(cfg.get("max_size", 1333))
    num_workers = int(cfg.get("num_workers", 4))
    num_classes = int(cfg.get("num_classes", 3))
    seed = int(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device"))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir = _resolve(Path(str(cfg["output_dir"])), root)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CocoDetectionTorchvision(
        _resolve(Path(str(cfg["coco_train"])), root),
        _resolve(Path(str(cfg["images_train"])), root),
        train=True,
        max_images=cfg.get("max_train_images"),
    )
    val_ds = CocoDetectionTorchvision(
        _resolve(Path(str(cfg["coco_val"])), root),
        _resolve(Path(str(cfg["images_val"])), root),
        train=False,
        max_images=cfg.get("max_val_images"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_faster_rcnn(
        num_classes=num_classes, min_size=min_size, max_size=max_size, pretrained=True
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = weights_dir / "best.pth"

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device, grad_accum)
        val_loss = _eval_loss(model, val_loader, device)
        lr_scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  saved best → {best_path}")

    last_path = weights_dir / "last.pth"
    torch.save(model.state_dict(), last_path)

    cfg_out = out_dir / "config.yaml"
    cfg_out.write_text(yaml.safe_dump(dict(cfg), sort_keys=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({"epochs": history, "best_val_loss": best_val}, indent=2), encoding="utf-8")
    (out_dir / "system_info.json").write_text(
        json.dumps({"system": _system_info(), "git_rev": _git_rev(root)}, indent=2),
        encoding="utf-8",
    )
    print(f"Training finished. best={best_path}")


if __name__ == "__main__":
    main()
