#!/usr/bin/env python3
"""Fine-tune RF-DETR on datasets/ants_coco (Roboflow COCO layout); mirror YOLO experiment metadata."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(p: Path, root: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


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

    import torch

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


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = sig.parameters
    # RF-DETR 1.6.0 exposes kwargs-driven APIs in some entrypoints; if the target
    # accepts **kwargs, keep all non-None keys instead of dropping required args.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {k: v for k, v in kwargs.items() if v is not None}
    return {k: v for k, v in kwargs.items() if k in params and v is not None}


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=str,
        default=str(root / "configs/expA005_ants_rfdetr.yaml"),
        help="YAML with model_class, dataset_dir, output_dir, train hyperparameters",
    )
    args = p.parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}

    try:
        rfdetr = importlib.import_module("rfdetr")
    except ImportError:
        print(
            "Install rfdetr: pip install rfdetr  (or pip install -e .[rfdetr])",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        importlib.import_module("albumentations")
    except ImportError:
        print(
            "Missing dependency: albumentations. "
            "Install RF-DETR extras via: pip install -e .[rfdetr]",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        importlib.import_module("faster_coco_eval")
    except ImportError:
        print(
            "Missing dependency: faster-coco-eval. "
            "Install RF-DETR extras via: pip install -e .[rfdetr]",
            file=sys.stderr,
        )
        sys.exit(1)

    model_name = str(cfg.get("model_class", "RFDETRSmall"))
    if not hasattr(rfdetr, model_name):
        print(f"Unknown rfdetr model_class: {model_name}", file=sys.stderr)
        sys.exit(1)
    ModelCls = getattr(rfdetr, model_name)

    dataset_dir = _resolve(Path(str(cfg["dataset_dir"])), root)
    output_dir = _resolve(Path(str(cfg["output_dir"])), root)
    resume = cfg.get("resume_checkpoint")
    resume_path = Path(str(resume)).expanduser().resolve() if resume else None

    train_json = dataset_dir / "train" / "_annotations.coco.json"
    valid_json = dataset_dir / "valid" / "_annotations.coco.json"
    if not train_json.is_file() or not valid_json.is_file():
        print(
            f"Missing RF-DETR COCO files under {dataset_dir}. "
            "Expected train/_annotations.coco.json and valid/_annotations.coco.json. "
            "Ants: scripts/datasets/prepare_ants_coco_rfdetr.py. "
            "Camponotus: scripts/datasets/prepare_camponotus_coco_rfdetr.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    init_kw: dict[str, Any] = {}
    if resume_path is not None and resume_path.is_file():
        init_kw["pretrain_weights"] = str(resume_path)
    model = ModelCls(**_filter_kwargs(ModelCls.__init__, init_kw))

    train_params = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "epochs": int(cfg.get("epochs", 30)),
        "batch_size": int(cfg.get("batch_size", 4)),
        "grad_accum_steps": int(cfg.get("grad_accum_steps", 4)),
        "lr": float(cfg.get("lr", 1e-4)),
    }
    if cfg.get("seed") is not None:
        train_params["seed"] = int(cfg["seed"])
    # Pass through optional keys supported by this rfdetr version
    for extra in (
        "device",
        "num_workers",
        "checkpoint_interval",
        "early_stopping",
        "early_stopping_patience",
        "tensorboard",
        "wandb",
    ):
        if cfg.get(extra) is not None:
            train_params[extra] = cfg[extra]

    train_call = _filter_kwargs(model.train, train_params)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {model_name} dataset_dir={dataset_dir} output_dir={output_dir}")
    model.train(**train_call)

    ckpt_total = output_dir / "checkpoint_best_total.pth"
    if not ckpt_total.is_file():
        for cand in (
            output_dir / "checkpoint_best_ema.pth",
            output_dir / "checkpoint_best_regular.pth",
            output_dir / "checkpoint.pth",
        ):
            if cand.is_file():
                ckpt_total = cand
                break

    wdir = output_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    best = wdir / "best.pth"
    if ckpt_total.is_file():
        shutil.copy2(ckpt_total, best)
        print(f"Copied weights → {best}")
    else:
        print(
            "Warning: no checkpoint_best_total.pth (or fallback) found after train.",
            file=sys.stderr,
        )

    resolved = {**cfg, "dataset_dir": str(dataset_dir), "output_dir": str(output_dir)}
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    meta = {
        **_system_info(),
        "git_rev": _git_rev(root),
        "model_class": model_name,
        "train_signature_filtered": list(train_call.keys()),
    }
    (output_dir / "system_info.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'config.yaml'} and system_info.json")


if __name__ == "__main__":
    main()
