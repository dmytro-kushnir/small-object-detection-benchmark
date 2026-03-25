#!/usr/bin/env python3
"""Rewrite COCO for CVAT import.

1) Default: remap category ids to **1..N** (CVAT ignores ``category_id: 0``; see
   https://github.com/cvat-ai/cvat/issues/4750 ).

2) **--collapse-to-single-label ant:** one CVAT rectangle label; every box gets ``category_id: 1``.

   Optional **--carry-state-attributes:** set each annotation's **``attributes.state``** from the
   source category name (``trophallaxis`` → ``--state-troph``, else ``--state-normal``). CVAT uses
   Datumaro for COCO import, which reads per-annotation **``attributes``** on bboxes (same idea as
   export). Your task must define attribute **``state``** on label **``ant``** with those values.

Training: ``prepare_camponotus_detection_dataset.py`` maps by **category name** / **state**.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from camponotus_common import read_json, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", type=str, required=True, help="Input COCO JSON")
    p.add_argument("--out", type=str, required=True, help="Output COCO JSON")
    p.add_argument(
        "--strip-score",
        action="store_true",
        help="Remove annotation ``score`` (strict COCO GT; try if CVAT still errors).",
    )
    p.add_argument(
        "--collapse-to-single-label",
        type=str,
        default="",
        metavar="NAME",
        help=(
            "Single CVAT rectangle label (e.g. ant). All annotations get category_id 1; "
            "categories list is one entry. For Pattern A + state attribute in CVAT."
        ),
    )
    p.add_argument(
        "--carry-state-attributes",
        action="store_true",
        help=(
            "With --collapse-to-single-label: add attributes.state from source category name "
            "(trophallaxis vs other). Requires matching attribute on the ant label in CVAT."
        ),
    )
    p.add_argument("--state-attr", type=str, default="state", help="Attribute name (default: state).")
    p.add_argument(
        "--state-normal",
        type=str,
        default="normal",
        help="Value for non-trophallaxis boxes (default: normal).",
    )
    p.add_argument(
        "--state-troph",
        type=str,
        default="trophallaxis",
        help="Value when source category name is trophallaxis (default: trophallaxis).",
    )
    args = p.parse_args()

    inp = Path(args.inp).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not inp.is_file():
        print(f"not found: {inp}", file=sys.stderr)
        sys.exit(1)

    coco: dict[str, Any] = deepcopy(read_json(inp))
    raw_cats = coco.get("categories")
    if not isinstance(raw_cats, list) or not raw_cats:
        print("COCO has no categories", file=sys.stderr)
        sys.exit(1)

    anns = coco.get("annotations")
    if not isinstance(anns, list):
        print("COCO has no annotations list", file=sys.stderr)
        sys.exit(1)

    if args.carry_state_attributes and not str(args.collapse_to_single_label).strip():
        print("--carry-state-attributes requires --collapse-to-single-label", file=sys.stderr)
        sys.exit(1)

    collapse = str(args.collapse_to_single_label).strip()
    if collapse:
        known = {int(c["id"]) for c in raw_cats}
        id_to_name = {int(c["id"]): str(c.get("name", "")).strip().lower() for c in raw_cats}
        trop_key = str(args.state_troph).strip().casefold()
        normal_val = str(args.state_normal)
        troph_val = str(args.state_troph)
        state_key = str(args.state_attr).strip()
        if not state_key:
            print("--state-attr must be non-empty", file=sys.stderr)
            sys.exit(1)

        for a in anns:
            if not isinstance(a, dict):
                continue
            oid = int(a["category_id"])
            if oid not in known:
                print(f"unknown category_id in annotation: {oid}", file=sys.stderr)
                sys.exit(1)
            if args.carry_state_attributes:
                cname = id_to_name.get(oid, "")
                is_troph = cname == trop_key or "trophallaxis" in cname
                state_val = troph_val if is_troph else normal_val
                prev = a.get("attributes")
                attrs: dict[str, Any] = dict(prev) if isinstance(prev, dict) else {}
                attrs[state_key] = state_val
                a["attributes"] = attrs
            a["category_id"] = 1
            if args.strip_score:
                a.pop("score", None)
        coco["categories"] = [
            {"id": 1, "name": collapse, "supercategory": "object"},
        ]
        write_json(out, coco)
        if args.carry_state_attributes:
            print(
                f"Wrote {out} (single label {collapse!r}; {state_key!r} = "
                f"{normal_val!r} | {troph_val!r} from source category names)"
            )
        else:
            print(
                f"Wrote {out} (single label {collapse!r}; no attributes — use --carry-state-attributes "
                f"to set {state_key!r} from trophallaxis vs ant)"
            )
        return

    ordered = sorted(raw_cats, key=lambda c: int(c["id"]))
    old_to_new = {int(c["id"]): i + 1 for i, c in enumerate(ordered)}
    new_cats: list[dict[str, Any]] = []
    for i, c in enumerate(ordered):
        nc = dict(c)
        nc["id"] = i + 1
        if "supercategory" not in nc:
            nc["supercategory"] = "object"
        new_cats.append(nc)

    for a in anns:
        if not isinstance(a, dict):
            continue
        oid = int(a["category_id"])
        if oid not in old_to_new:
            print(f"unknown category_id in annotation: {oid}", file=sys.stderr)
            sys.exit(1)
        a["category_id"] = old_to_new[oid]
        if args.strip_score:
            a.pop("score", None)

    coco["categories"] = new_cats
    write_json(out, coco)
    print(f"Wrote {out} (category map: {old_to_new})")


if __name__ == "__main__":
    main()
