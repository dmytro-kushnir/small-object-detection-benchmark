#!/usr/bin/env python3
"""Create sequence-safe train/val/test split manifests for Camponotus dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from camponotus_common import seeded_shuffle, write_json


def _list_sequence_dirs(in_situ_root: Path) -> list[Path]:
    return sorted(p for p in in_situ_root.iterdir() if p.is_dir() and p.name.startswith("seq_"))


def _list_external_images(external_images_dir: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(str(p.resolve()) for p in external_images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _seq_images(seq_dir: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(str(p.resolve()) for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _split_sequence_names(
    names: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    shuffled = seeded_shuffle(names, seed)
    n = len(shuffled)
    if n == 0:
        return [], [], []
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1 if n >= 3 else max(0, n - 2)), n)
    n_val = min(max(n_val, 1 if n >= 3 else (1 if n >= 2 else 0)), max(0, n - n_train))
    n_test = max(0, n - n_train - n_val)
    if n_test == 0 and n >= 3:
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
        n_test = 1
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val : n_train + n_val + n_test])
    return train, val, test


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-situ-root", type=str, default="datasets/camponotus_raw/in_situ")
    p.add_argument("--external-images-dir", type=str, default="datasets/camponotus_raw/external/images")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="datasets/camponotus_processed/splits.json")
    args = p.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    in_situ_root = Path(args.in_situ_root).expanduser().resolve()
    external_dir = Path(args.external_images_dir).expanduser().resolve()
    if not in_situ_root.is_dir():
        raise FileNotFoundError(f"in-situ root not found: {in_situ_root}")
    if not external_dir.is_dir():
        raise FileNotFoundError(f"external images dir not found: {external_dir}")

    seq_dirs = _list_sequence_dirs(in_situ_root)
    seq_names = [p.name for p in seq_dirs]
    train_seq, val_seq, test_seq = _split_sequence_names(
        names=seq_names,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    seq_map = {p.name: p for p in seq_dirs}
    train_images = [img for s in train_seq for img in _seq_images(seq_map[s])]
    val_images = [img for s in val_seq for img in _seq_images(seq_map[s])]
    test_images = [img for s in test_seq for img in _seq_images(seq_map[s])]
    ext_images = _list_external_images(external_dir)
    train_images.extend(ext_images)  # external defaults to train only

    payload: dict[str, Any] = {
        "schema_version": 1,
        "policy": {
            "split_by_sequence": True,
            "val_test_only_in_situ": True,
            "external_to_train_only": True,
        },
        "ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "seed": int(args.seed),
        "in_situ_sequences": {
            "train": train_seq,
            "val": val_seq,
            "test": test_seq,
        },
        "images": {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        },
        "counts": {
            "sequences_total": len(seq_dirs),
            "sequences_train": len(train_seq),
            "sequences_val": len(val_seq),
            "sequences_test": len(test_seq),
            "images_train": len(train_images),
            "images_val": len(val_images),
            "images_test": len(test_images),
            "images_external_train": len(ext_images),
        },
    }
    out_path = Path(args.out).expanduser().resolve()
    write_json(out_path, payload)
    print(f"Wrote split manifest: {out_path}")
    print(
        f"Sequences train/val/test: {len(train_seq)}/{len(val_seq)}/{len(test_seq)}; "
        f"images train/val/test: {len(train_images)}/{len(val_images)}/{len(test_images)}"
    )


if __name__ == "__main__":
    main()
