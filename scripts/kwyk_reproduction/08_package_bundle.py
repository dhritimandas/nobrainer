#!/usr/bin/env python
"""Package a converted kwyk checkpoint as a MONAI bundle.

``nobrainer.datasets.convert_kwyk`` writes a bare state_dict (``.pth``) plus
a ``.provenance.json`` sidecar -- not a ``Segmentation.save()`` directory
(``model.pth`` + ``croissant.json``), which is what
``nobrainer export bundle`` requires as input. This script bridges that gap:
it builds a ``Segmentation`` estimator from the converted checkpoint's own
provenance (architecture args, n_classes) plus kwyk's fixed block shape,
and calls ``.save()`` to produce a directory the CLI can consume directly.

Usage:
    python 08_package_bundle.py \\
        --pth kwyk_map.pth \\
        --parity-report results/parity/parity_report.json \\
        --output-dir checkpoints/kwyk_map

    # then:
    uv run nobrainer export bundle checkpoints/kwyk_map bundles/kwyk_map \\
        --allow-stochastic --name "kwyk MAP (all_50_wn)" ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from utils import setup_logging

log = setup_logging(__name__)

# kwyk's SavedModel has a fixed (-1, 32, 32, 32, 1) input signature (verified
# against a live neuronets/kwyk container -- see docs/kwyk_parity_report.md).
# convert_kwyk.py's provenance sidecar does not record this (it is not a
# constructor argument), so it is hardcoded here, matching
# scripts/kwyk_reproduction/07_parity_kwyk.py's _BLOCK_SHAPE.
_KWYK_BLOCK_SHAPE = (32, 32, 32)

# Keys in convert_kwyk.py's provenance JSON that are kwyk_meshnet
# constructor arguments (excluding n_classes, which Segmentation._build_model
# supplies separately from n_classes_).
_MODEL_ARG_KEYS = ("filters", "receptive_field", "dropout_type", "bias")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pth", type=str, required=True, help="Converted checkpoint (.pth)."
    )
    parser.add_argument(
        "--provenance",
        type=str,
        default=None,
        help="Path to the .provenance.json sidecar (default: <pth stem>.provenance.json).",
    )
    parser.add_argument(
        "--parity-report",
        type=str,
        default=None,
        help=(
            "Optional parity_report.json (from 07_parity_kwyk.py). If given, "
            "the pass status of the stage matching --parity-stage is logged "
            "(warning, not a hard failure, if it's false), so an unvalidated "
            "or partially-validated checkpoint is never packaged silently as "
            "if fully verified. The report's top-level result.pass is NOT "
            "used directly -- it blends Stage A (MAP) and Stage B (SSD), so "
            "checking it would misreport one model's status using the "
            "other's failure."
        ),
    )
    parser.add_argument(
        "--parity-stage",
        choices=("stage_a_deterministic", "stage_b_mc"),
        default="stage_a_deterministic",
        help="Which stage's aggregate.pass in --parity-report describes this checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write (must not already exist) -- model.pth + croissant.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pth_path = Path(args.pth)
    provenance_path = (
        Path(args.provenance)
        if args.provenance
        else pth_path.with_name(pth_path.stem + ".provenance.json")
    )
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")

    provenance = json.loads(provenance_path.read_text())
    base_model = provenance["model_registry_name"]
    n_classes = provenance["n_classes"]
    model_args = {k: provenance[k] for k in _MODEL_ARG_KEYS if k in provenance}
    log.info(
        "Packaging %s: base_model=%s n_classes=%d model_args=%s",
        pth_path,
        base_model,
        n_classes,
        model_args,
    )

    if args.parity_report:
        report = json.loads(Path(args.parity_report).read_text())
        stage = report.get(args.parity_stage)
        stage_pass = stage.get("aggregate", {}).get("pass") if stage else None
        if stage_pass is None:
            log.warning(
                "Parity report %s has no '%s' section -- this checkpoint's "
                "parity status is unknown, not verified.",
                args.parity_report,
                args.parity_stage,
            )
        elif stage_pass is False:
            log.warning(
                "Parity report %s: %s.aggregate.pass=false -- packaging "
                "anyway (not a hard block), but this checkpoint is NOT "
                "parity-validated. See the report before treating the "
                "bundle as verified.",
                args.parity_report,
                args.parity_stage,
            )
        else:
            log.info(
                "Parity report %s: %s.aggregate.pass=true.",
                args.parity_report,
                args.parity_stage,
            )

    from nobrainer.models import get as get_model
    from nobrainer.processing.segmentation import Segmentation

    model = get_model(base_model)(n_classes=n_classes, **model_args)
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    seg = Segmentation(base_model=base_model, model_args=model_args)
    seg.model_ = model
    seg.n_classes_ = n_classes
    seg.block_shape_ = _KWYK_BLOCK_SHAPE
    seg.save(output_dir)
    log.info("Wrote Segmentation directory to %s", output_dir)


if __name__ == "__main__":
    main()
