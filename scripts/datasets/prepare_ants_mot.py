#!/usr/bin/env python3
"""MOT-format ant annotations → YOLO dataset + COCO val/train JSON + analysis + manifest."""

from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

try:
    import hydra
    from hydra.utils import get_original_cwd
except ImportError:  # pragma: no cover
    hydra = None  # type: ignore
    get_original_cwd = None  # type: ignore


def _load_prepare_dataset_module() -> Any:
    """Load sibling prepare_dataset.py (no package; avoids bare `import prepare_dataset`)."""
    mod_path = Path(__file__).resolve().parent / "prepare_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_dataset", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prepare_dataset from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve(p: str | Path) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = Path.cwd()
    if get_original_cwd is not None:
        try:
            base = Path(get_original_cwd())
        except Exception:
            pass
    return (base / path).resolve()


def _seq_key(dataset_root: Path, seq_dir: Path) -> str:
    rel = seq_dir.resolve().relative_to(dataset_root.resolve())
    s = str(rel).replace("/", "_").replace("\\", "_")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_") or "seq"


def _discover_mot_sequences(
    dataset_root: Path, gt_subdir: str, gt_filename: str
) -> list[tuple[Path, Path]]:
    out: list[tuple[Path, Path]] = []
    for gt_path in sorted(dataset_root.rglob(gt_filename)):
        if gt_path.parent.name != gt_subdir:
            continue
        seq_dir = gt_path.parent.parent
        if not seq_dir.is_dir():
            continue
        img_dir = seq_dir / "img1"
        if not img_dir.is_dir():
            alt = seq_dir / "img"
            if alt.is_dir():
                pass
        out.append((seq_dir.resolve(), gt_path.resolve()))
    return sorted(out, key=lambda x: str(x[0]))


def _parse_mot_line(
    line: str,
    delimiter: str,
    skip_neg_conf: bool,
) -> tuple[int, float, float, float, float] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if delimiter == "comma":
        parts = [p.strip() for p in line.split(",")]
    elif delimiter == "space":
        parts = line.split()
    else:
        parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
    if len(parts) < 6:
        return None
    try:
        fid = int(float(parts[0]))
        x, y, w, h = map(float, parts[2:6])
    except (ValueError, TypeError):
        return None
    if skip_neg_conf and len(parts) >= 8:
        try:
            conf = float(parts[6])
            if conf < 0:
                return None
        except (ValueError, TypeError):
            pass
    if w <= 0 or h <= 0:
        return None
    return fid, x, y, w, h


def _load_gt_by_frame(gt_path: Path, delimiter: str, skip_neg_conf: bool) -> dict[int, list[tuple[float, float, float, float]]]:
    by_f: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            parsed = _parse_mot_line(line, delimiter, skip_neg_conf)
            if parsed is None:
                continue
            fid, x, y, w, h = parsed
            by_f[fid].append((x, y, w, h))
    return by_f


def _frame_image_path(
    seq_dir: Path, img_subdir: str, frame_id: int, frame_digits: int, image_ext: str
) -> Path:
    name = f"{frame_id:0{frame_digits}d}{image_ext}"
    p = seq_dir / img_subdir / name
    if p.is_file():
        return p
    p2 = seq_dir / "img" / name
    if p2.is_file():
        return p2
    return seq_dir / img_subdir / name


def _xywh_to_yolo_line(
    x: float,
    y: float,
    w: float,
    h: float,
    im_w: int,
    im_h: int,
    class_id: int,
) -> str | None:
    x1, y1, x2, y2 = x, y, x + w, y + h
    x1 = max(0.0, min(float(im_w), x1))
    y1 = max(0.0, min(float(im_h), y1))
    x2 = max(0.0, min(float(im_w), x2))
    y2 = max(0.0, min(float(im_h), y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 1 or bh < 1:
        return None
    cx = (x1 + x2) / 2.0 / im_w
    cy = (y1 + y2) / 2.0 / im_h
    nw = bw / im_w
    nh = bh / im_h
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


@dataclass
class FrameJob:
    seq_key: str
    frame_id: int
    src_image: Path
    dst_name: str
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    split: str = "train"


def _frame_ids_from_image_dir(seq_dir: Path, img_sub: str) -> set[int]:
    """Numeric stems under seq_dir/img_sub (any common image ext)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_root = seq_dir / img_sub
    if not img_root.is_dir() and (seq_dir / "img").is_dir():
        img_root = seq_dir / "img"
    if not img_root.is_dir():
        return set()
    out: set[int] = set()
    for f in img_root.iterdir():
        if f.suffix.lower() not in exts:
            continue
        try:
            out.add(int(f.stem))
        except ValueError:
            continue
    return out


def _collect_frame_jobs(
    dataset_root: Path,
    sequences: list[tuple[Path, Path]],
    cfg: DictConfig,
) -> list[FrameJob]:
    delim = str(cfg.delimiter)
    if delim not in ("auto", "comma", "space"):
        delim = "auto"
    skip_neg = bool(cfg.skip_mot_confidence_neg_one)
    img_sub = str(cfg.img_subdir)
    ext = str(cfg.image_ext)
    if not ext.startswith("."):
        ext = "." + ext
    digits = int(cfg.frame_digits)
    jobs: list[FrameJob] = []
    for seq_dir, gt_path in sequences:
        sk = _seq_key(dataset_root, seq_dir)
        by_f = _load_gt_by_frame(gt_path, delim, skip_neg)
        from_disk = _frame_ids_from_image_dir(seq_dir, img_sub)
        fids = sorted(from_disk | set(by_f.keys()))
        if not fids:
            continue
        for fid in fids:
            ip = _frame_image_path(seq_dir, img_sub, fid, digits, ext)
            if not ip.is_file():
                for alt in (".jpg", ".jpeg", ".png", ".bmp"):
                    ip2 = _frame_image_path(seq_dir, img_sub, fid, digits, alt)
                    if ip2.is_file():
                        ip = ip2
                        break
            if not ip.is_file():
                print(f"Warning: missing image for {sk} frame {fid}: {ip}", file=sys.stderr)
                continue
            dst_name = f"{sk}__f{fid:0{digits}d}{ip.suffix.lower()}"
            jobs.append(
                FrameJob(
                    seq_key=sk,
                    frame_id=fid,
                    src_image=ip,
                    dst_name=dst_name,
                    boxes=list(by_f.get(fid, [])),
                    split="train",
                )
            )
    return jobs


def _split_temporal_per_sequence(jobs: list[FrameJob], train_ratio: float) -> None:
    tr = float(train_ratio)
    by_seq: dict[str, list[FrameJob]] = defaultdict(list)
    for j in jobs:
        by_seq[j.seq_key].append(j)
    for seq_jobs in by_seq.values():
        seq_jobs.sort(key=lambda x: x.frame_id)
        n = len(seq_jobs)
        if n == 0:
            continue
        n_train = int(math.floor(n * tr))
        if n >= 2:
            n_train = max(1, min(n - 1, n_train))
            for i, j in enumerate(seq_jobs):
                j.split = "train" if i < n_train else "val"
        else:
            seq_jobs[0].split = "train"


def _split_temporal_global(jobs: list[FrameJob], train_ratio: float) -> None:
    tr = float(train_ratio)
    jobs.sort(key=lambda j: (j.seq_key, j.frame_id))
    n = len(jobs)
    if n == 0:
        return
    n_train = int(math.floor(n * tr))
    if n >= 2:
        n_train = max(1, min(n - 1, n_train))
        for i, j in enumerate(jobs):
            j.split = "train" if i < n_train else "val"
    else:
        jobs[0].split = "train"


def _ensure_non_empty_val(jobs: list[FrameJob]) -> None:
    if sum(1 for j in jobs if j.split == "val") == 0 and len(jobs) > 1:
        jobs[-1].split = "val"


def _write_yolo_dataset(
    output_root: Path,
    jobs: list[FrameJob],
    cfg: DictConfig,
) -> None:
    mode = str(cfg.link_mode)
    if mode not in ("symlink", "copy"):
        mode = "symlink"
    cls_id = int(cfg.class_id_yolo)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for job in tqdm(jobs, desc="export_frames"):
        split = job.split
        dst_img = output_root / "images" / split / job.dst_name
        _link_or_copy(job.src_image, dst_img, mode)
        im = cv2.imread(str(dst_img))
        if im is None:
            print(f"Warning: could not read {dst_img}", file=sys.stderr)
            continue
        h, w = im.shape[:2]
        lines: list[str] = []
        for x, y, bw, bh in job.boxes:
            ln = _xywh_to_yolo_line(x, y, bw, bh, w, h, cls_id)
            if ln:
                lines.append(ln)
        lbl = output_root / "labels" / split / Path(job.dst_name).with_suffix(".txt").name
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_dataset_yaml(output_root: Path) -> Path:
    p = output_root / "dataset.yaml"
    cfg = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["ant"],
    }
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return p


def _compute_analysis(output_root: Path, class_id: int) -> dict[str, Any]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    areas: list[float] = []
    sides: list[float] = []
    norm_areas: list[float] = []
    n_images = {"train": 0, "val": 0}
    n_ann = {"train": 0, "val": 0}
    objs_per_frame: list[int] = []
    small_n = med_n = large_n = 0
    for split in ("train", "val"):
        img_dir = output_root / "images" / split
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            n_images[split] += 1
            lbl = output_root / "labels" / split / img_path.with_suffix(".txt").name
            k = 0
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            ih, iw = im.shape[:2]
            if lbl.is_file():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    xc, yc, nw, nh = map(float, parts[1:5])
                    bw = nw * iw
                    bh = nh * ih
                    area = bw * bh
                    side = math.sqrt(max(area, 1e-6))
                    areas.append(area)
                    sides.append(side)
                    norm_areas.append(nw * nh)
                    k += 1
                    if side < 32:
                        small_n += 1
                    elif side <= 96:
                        med_n += 1
                    else:
                        large_n += 1
            n_ann[split] += k
            objs_per_frame.append(k)
    total_ann = small_n + med_n + large_n
    frac_small = (small_n / total_ann) if total_ann else 0.0
    mean_opf = sum(objs_per_frame) / len(objs_per_frame) if objs_per_frame else 0.0
    var = (
        sum((x - mean_opf) ** 2 for x in objs_per_frame) / len(objs_per_frame)
        if objs_per_frame
        else 0.0
    )
    return {
        "class_id_yolo": class_id,
        "coco_size_definition": "side = sqrt(bbox_area) in pixels; small<32, medium<=96, large>96",
        "images": n_images,
        "annotations_total": n_ann["train"] + n_ann["val"],
        "annotations_by_split": n_ann,
        "mean_objects_per_frame": mean_opf,
        "std_objects_per_frame": math.sqrt(var),
        "bbox_area_px_mean": sum(areas) / len(areas) if areas else None,
        "bbox_side_px_mean": sum(sides) / len(sides) if sides else None,
        "normalized_area_mean": sum(norm_areas) / len(norm_areas) if norm_areas else None,
        "count_small_medium_large": {"small": small_n, "medium": med_n, "large": large_n},
        "fraction_small_of_annotations": frac_small,
    }


def run_prepare(cfg: DictConfig) -> None:
    if hydra is None:
        raise RuntimeError("hydra-core is required")

    _pd = _load_prepare_dataset_module()
    yolo_to_coco_in_memory_for_split = _pd.yolo_to_coco_in_memory_for_split

    dataset_root = _resolve(cfg.dataset_root)
    output_root = _resolve(cfg.output_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    layout = str(cfg.layout)
    if layout != "mot_sequence":
        raise NotImplementedError(
            f"layout={layout!r} not implemented; use mot_sequence (…/gt/gt.txt + img1/)."
        )

    sequences = _discover_mot_sequences(
        dataset_root, str(cfg.gt_subdir), str(cfg.gt_filename)
    )
    if not sequences:
        raise FileNotFoundError(
            f"No MOT sequences under {dataset_root}: expected **/{cfg.gt_subdir}/{cfg.gt_filename}"
        )

    strategy = str(cfg.split_strategy)
    jobs = _collect_frame_jobs(dataset_root, sequences, cfg)
    if strategy == "per_sequence":
        _split_temporal_per_sequence(jobs, float(cfg.train_ratio))
    elif strategy == "global_sorted":
        _split_temporal_global(jobs, float(cfg.train_ratio))
    else:
        raise ValueError(f"Unknown split_strategy: {strategy}")
    _ensure_non_empty_val(jobs)

    if not jobs:
        raise RuntimeError("No frames exported; check frame_digits, image_ext, and image paths.")

    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_yolo_dataset(output_root, jobs, cfg)
    data_yaml = _write_dataset_yaml(output_root)

    ann_dir = output_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        coco = yolo_to_coco_in_memory_for_split(output_root, data_yaml, split)
        out_p = ann_dir / f"instances_{split}.json"
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(coco, f)

    analysis = _compute_analysis(output_root, int(cfg.class_id_yolo))
    (output_root / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )

    git_rev: str | None = None
    repo_root = Path(__file__).resolve().parents[2]
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    manifest = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "split_strategy": strategy,
        "train_ratio": float(cfg.train_ratio),
        "val_ratio": float(cfg.val_ratio),
        "seed": int(cfg.seed),
        "link_mode": str(cfg.link_mode),
        "layout": layout,
        "sequences": [str(s[0]) for s in sequences],
        "frame_jobs": len(jobs),
        "counts_train_val": {
            "train": sum(1 for j in jobs if j.split == "train"),
            "val": sum(1 for j in jobs if j.split == "val"),
        },
        "git_rev": git_rev,
        "dataset_yaml": str(data_yaml),
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
    }
    (output_root / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Done. YOLO + COCO under {output_root}")


def main() -> None:
    if hydra is None:
        print("hydra-core is required", file=sys.stderr)
        sys.exit(1)


if hydra is not None:

    @hydra.main(
        version_base=None,
        config_path="../../configs",
        config_name="datasets/ants_mot_prepare",
    )
    def _cli(cfg: DictConfig) -> None:
        run_prepare(cfg)


if __name__ == "__main__":
    if hydra is None:
        main()
    else:
        _cli()
