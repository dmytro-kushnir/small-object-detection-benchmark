"""Portable path strings for metrics, manifests, and docs committed to the repo.

* Paths under the repository root -> POSIX relative (e.g. ``experiments/results/x.json``).
* Other paths that contain a ``datasets`` path segment (external video/COCO trees) ->
  ``<DATASETS_ROOT>/...`` so clones can set one env or config root.
* Anything else stays absolute (last resort; prefer passing repo-local paths in scripts).
"""

from __future__ import annotations

from pathlib import Path


def path_for_artifact(path: Path, repo_root: Path) -> str:
    p = path.expanduser().resolve()
    root = repo_root.resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        parts = p.parts
        if "datasets" in parts:
            i = parts.index("datasets")
            tail = parts[i + 1 :]
            if not tail:
                return "<DATASETS_ROOT>"
            sub = Path(*tail)
            return f"<DATASETS_ROOT>/{sub.as_posix()}"
        return p.as_posix()
