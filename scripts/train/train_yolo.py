#!/usr/bin/env python3
"""Train Ultralytics YOLO26 (or other YOLO .pt) with Hydra; artifacts under experiments/yolo/<run_id>/."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

try:
    import hydra
    from hydra.utils import get_original_cwd
except ImportError:  # pragma: no cover
    hydra = None  # type: ignore
    get_original_cwd = None  # type: ignore


def _resolve_path_str(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    base = Path.cwd()
    if get_original_cwd is not None:
        try:
            base = Path(get_original_cwd())
        except Exception:
            pass
    return (base / p).resolve()


def _run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


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


def _read_metrics_csv(csv_path: Path) -> dict[str, Any]:
    if not csv_path.is_file():
        return {}
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return {}
    headers = lines[0].split(",")
    last = lines[-1].split(",")
    return dict(zip(headers, last))


def _gather_ultralytics_metrics(save_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"save_dir": str(save_dir)}
    results_csv = save_dir / "results.csv"
    out["results_last_row"] = _read_metrics_csv(results_csv)
    args_yaml = save_dir / "args.yaml"
    if args_yaml.is_file():
        import yaml

        with open(args_yaml, encoding="utf-8") as f:
            out["args_yaml"] = yaml.safe_load(f)
    return out


def run_train(cfg: DictConfig) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(
            "ultralytics is required (install with pip install -r requirements.txt). "
            "For YOLO26 weights, use ultralytics>=8.4.0.",
            file=sys.stderr,
        )
        raise e

    project = _resolve_path_str(cfg.project)
    name = cfg.name if cfg.name else _run_id()
    data = _resolve_path_str(cfg.data)
    if not data.is_file():
        raise FileNotFoundError(f"dataset yaml not found: {data}")

    train_kw: dict[str, Any] = {
        "data": str(data),
        "epochs": int(cfg.epochs),
        "imgsz": int(cfg.imgsz),
        "batch": int(cfg.batch),
        "workers": int(cfg.workers),
        "project": str(project),
        "name": str(name),
        "patience": int(cfg.patience),
        "save": bool(cfg.save),
        "plots": bool(cfg.plots),
        "verbose": bool(cfg.verbose),
        "seed": int(cfg.seed),
        "exist_ok": bool(cfg.exist_ok),
    }
    if cfg.device is not None:
        train_kw["device"] = cfg.device
    if cfg.optimizer.lr0 is not None:
        train_kw["lr0"] = float(cfg.optimizer.lr0)

    model = YOLO(str(cfg.model))
    model.train(**train_kw)

    trainer = getattr(model, "trainer", None)
    if trainer is not None and getattr(trainer, "save_dir", None):
        save_dir = Path(trainer.save_dir).resolve()
    else:
        save_dir = (project / str(name)).resolve()
    bundle_dir = save_dir
    OmegaConf.save(cfg, bundle_dir / "config.yaml")
    metrics = _gather_ultralytics_metrics(save_dir)
    (bundle_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    si = _system_info()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        si["git_rev"] = rev
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    (bundle_dir / "system_info.json").write_text(
        json.dumps(si, indent=2), encoding="utf-8"
    )
    print(f"Run complete. Artifacts: {bundle_dir}")


def main() -> None:
    if hydra is None:
        print("hydra-core is required", file=sys.stderr)
        sys.exit(1)


if hydra is not None:

    @hydra.main(
        version_base=None,
        config_path="../../configs",
        config_name="train/yolo",
    )
    def _cli(cfg: DictConfig) -> None:
        run_train(cfg)


if __name__ == "__main__":
    if hydra is None:
        main()
    else:
        _cli()
