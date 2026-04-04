#!/usr/bin/env python3
"""Rewrite COCO ``images[].file_name`` to paths relative to ``--raw-root`` by scanning the tree.

``--raw-root`` is usually either:

- ``datasets/camponotus_raw`` with ``in_situ/seq_*/*.jpg`` (repo-local frames), or
- the **CVAT export bundle root** — the directory that contains ``images/`` next to
  ``annotations/*.json``.

Resolution order for each ``images[]`` row:

1. If ``raw_root / file_name`` already exists, keep ``file_name`` (normalized).
2. Else map **basename** → file(s) under ``raw-root``. If exactly one match, use it.
3. If multiple matches, **narrow** using directory hints from the original ``file_name``
   (e.g. ``seq_camponotus_017/000020.jpg`` → only paths containing segment ``seq_camponotus_017``).
4. If still ambiguous, retry the index with ``--exclude-path-substring`` values (or the
   built-in retry excluding ``default/in_situ``, which skips duplicate frame mirrors that
   often sit next to a flat CVAT ``images/`` export).

Then ``prepare_camponotus_detection_dataset.py`` can open ``raw_root / file_name`` for each image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _norm_fn(s: str) -> str:
    s = str(s).strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s


def build_index(
    raw_root: Path,
    exts: set[str],
    exclude_substrings: tuple[str, ...],
) -> dict[str, list[str]]:
    by_base: dict[str, list[str]] = {}
    root_r = raw_root.resolve()
    for path in raw_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        rel = path.resolve().relative_to(root_r).as_posix()
        if any(sub in rel for sub in exclude_substrings):
            continue
        by_base.setdefault(path.name, []).append(rel)
    for k in by_base:
        by_base[k] = sorted(set(by_base[k]))
    return by_base


def _narrow_by_hints(candidates: list[str], fn: str) -> list[str]:
    parts = [p for p in Path(fn).parts if p not in (".", "")]
    if len(parts) <= 1:
        return candidates
    hints = parts[:-1]
    out = list(candidates)
    for h in hints:
        out = [c for c in out if h in Path(c).parts]
        if len(out) <= 1:
            break
    return out


def resolve_images(
    images: list,
    raw_root: Path,
    by_base: dict[str, list[str]],
) -> tuple[list[str], list[tuple[str, str, list[str]]], list[str]]:
    """Return (missing_basenames, ambiguous_triples, warning_lines)."""
    missing: list[str] = []
    ambiguous: list[tuple[str, str, list[str]]] = []
    warnings: list[str] = []

    root_r = raw_root.resolve()
    for im in images:
        if not isinstance(im, dict):
            continue
        fn_raw = im.get("file_name", "")
        fn = _norm_fn(str(fn_raw))
        if not fn:
            missing.append("<empty>")
            continue

        direct = (root_r / fn).resolve()
        try:
            direct.relative_to(root_r)
        except ValueError:
            direct = None  # type: ignore[assignment]
        if direct is not None and direct.is_file():
            im["file_name"] = Path(fn).as_posix()
            continue

        base = Path(fn).name
        candidates = list(by_base.get(base, []))
        if not candidates:
            missing.append(base)
            continue

        narrowed = _narrow_by_hints(candidates, fn)
        if len(narrowed) == 1:
            im["file_name"] = narrowed[0]
            if len(candidates) > 1:
                warnings.append(
                    f"{base}: chose 1 of {len(candidates)} via path hints (COCO had {fn!r})"
                )
            continue
        if len(narrowed) > 1:
            ambiguous.append((base, fn, narrowed))
            continue

        if len(candidates) == 1:
            im["file_name"] = candidates[0]
            continue

        ambiguous.append((base, fn, candidates))

    return missing, ambiguous, warnings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", type=str, required=True, help="Input COCO JSON (e.g. CVAT export)")
    p.add_argument(
        "--raw-root",
        type=str,
        default="datasets/camponotus_raw",
        help="Root whose relative paths appear in output file_name (repo: camponotus_raw)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON (default: overwrite --coco if empty)",
    )
    p.add_argument(
        "--exclude-path-substring",
        type=str,
        action="append",
        default=[],
        metavar="SUBSTR",
        help=(
            "Skip image files whose path relative to raw-root contains this substring "
            "(repeatable). Use e.g. default/in_situ to ignore mirrored seq_* extractions "
            "next to a flat CVAT images/ folder."
        ),
    )
    args = p.parse_args()

    coco_path = Path(args.coco).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else coco_path

    if not coco_path.is_file():
        print(f"COCO not found: {coco_path}", file=sys.stderr)
        sys.exit(1)
    if not raw_root.is_dir():
        print(f"raw root not found: {raw_root}", file=sys.stderr)
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    user_excludes = tuple(str(x) for x in (args.exclude_path_substring or []) if str(x).strip())

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = coco.get("images")
    if not isinstance(images, list):
        print("COCO has no images list", file=sys.stderr)
        sys.exit(1)

    def try_resolve(excludes: tuple[str, ...]):
        by_base = build_index(raw_root, exts, excludes)
        return resolve_images(images, raw_root, by_base)

    missing, ambiguous, warnings = try_resolve(user_excludes)

    auto_note = ""
    if ambiguous or missing:
        if "default/in_situ" not in user_excludes:
            m2, a2, w2 = try_resolve(user_excludes + ("default/in_situ",))
            if not m2 and not a2:
                missing, ambiguous, warnings = m2, a2, w2
                user_excludes = user_excludes + ("default/in_situ",)
                auto_note = (
                    "Note: indexed images while excluding paths containing "
                    "'default/in_situ' (duplicate frame mirror next to CVAT export).\n"
                )

    if warnings:
        for line in warnings[:5]:
            print(line, file=sys.stderr)
        if len(warnings) > 5:
            print(f"... and {len(warnings) - 5} more basename disambiguation(s)", file=sys.stderr)

    if ambiguous:
        print(auto_note, end="", file=sys.stderr)
        print("Ambiguous basename → multiple files under raw-root after hints:", file=sys.stderr)
        for b, fn, c in ambiguous[:25]:
            preview = c[:3]
            more = f" (+{len(c) - 3} more)" if len(c) > 3 else ""
            print(f"  {b} (from {fn!r}): {len(c)} choices, e.g. {preview}{more}", file=sys.stderr)
        if len(ambiguous) > 25:
            print(f"  ... and {len(ambiguous) - 25} more", file=sys.stderr)
        print(
            "Fix: pass --exclude-path-substring (e.g. default/in_situ), or rename frames so "
            "each basename is unique, or set CVAT/COCO file_name to a full relative path.",
            file=sys.stderr,
        )
        sys.exit(1)

    if missing:
        print(auto_note, end="", file=sys.stderr)
        uniq = sorted(set(missing))
        print(
            f"Missing on disk under {raw_root} ({len(missing)} images), "
            f"e.g. basenames: {uniq[:15]}{'…' if len(uniq) > 15 else ''}",
            file=sys.stderr,
        )
        sys.exit(1)

    if auto_note:
        print(auto_note.strip())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(images)} images, file_name → relative to {raw_root})")


if __name__ == "__main__":
    main()
