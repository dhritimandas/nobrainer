#!/usr/bin/env python
"""Append kwyk-specific provenance and caveats to an exported bundle's README.

``nobrainer export bundle`` (nobrainer/export/bundle.py) writes a generic
``docs/README.md`` -- disclaimer, architecture, patch shape, a generic run
command. It does not know about kwyk's citation or parity-validation
history, and its own stochasticity self-check is a generic ``net(probe)``
call with no ``mc`` argument, which reports the kwyk MAP checkpoint as
"stochastic" even though the network's actual intended deterministic path
requires ``mc=False`` explicitly -- something this script documents rather
than silently leaves implied.

Usage:
    python 09_annotate_bundle_readme.py \\
        --bundle-dir bundles/kwyk_map \\
        --parity-report results/parity/parity_report.json \\
        --parity-stage stage_a_deterministic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=str, required=True)
    parser.add_argument("--parity-report", type=str, default=None)
    parser.add_argument(
        "--parity-stage",
        choices=("stage_a_deterministic", "stage_b_mc"),
        default="stage_a_deterministic",
    )
    return parser.parse_args()


def _parity_section(report_path: str | None, stage_key: str) -> str:
    if not report_path:
        return ""
    report = json.loads(Path(report_path).read_text())
    stage = report.get(stage_key, {})
    agg = stage.get("aggregate", {})
    volumes = stage.get("volumes", [])
    lines = [
        "## Parity validation",
        "",
        f"Checked against a live `neuronets/kwyk` container "
        f"(`{report.get('environment', {}).get('container_image', 'neuronets/kwyk')}`) "
        f"via `scripts/kwyk_reproduction/07_parity_kwyk.py`, stage `{stage_key}`. "
        f"Aggregate result: **{'PASS' if agg.get('pass') else 'FAIL/PARTIAL'}**.",
        "",
    ]
    for v in volumes:
        if stage_key == "stage_a_deterministic":
            lines.append(
                f"- `{v['volume_id']}`: max\\|logit diff\\| = {v['max_abs_logit_diff']:.2e}, "
                f"dice_mean = {v['dice_mean']:.4f}, "
                f"voxel_agreement = {v['voxel_agreement']:.4f}"
            )
        else:
            lines.append(
                f"- `{v['volume_id']}`: mean_label_dice = {v['mean_label_dice']:.4f}, "
                f"variance_pearson_r = {v['variance_pearson_r']:.4f}, "
                f"entropy_pearson_r = {v['entropy_pearson_r']:.4f}"
            )
    for note in report.get("notes", []):
        lines.append(f"\n> {note}")
    lines.append(
        "\nSee `docs/kwyk_parity_report.md` in the nobrainer repository for "
        "the full methodology, thresholds, and scope caveats."
    )
    return "\n".join(lines) + "\n"


_DETERMINISM_CAVEAT = """## Determinism caveat

This bundle's self-check (a plain forward pass with no `mc` argument) reports
the network as stochastic, and `configs/inference.json`'s `network_def` calls
it the same way -- so running this bundle via `python -m monai.bundle run`
samples the network's variational weight distribution (`mc=True`), it does
**not** use the deterministic mean-weights path. The MAP checkpoint's parity
validation (below) was performed with `mc=False` explicitly, matching
`nobrainer.prediction.predict()`'s own behavior for any model that declares
`mc` support. To reproduce the validated deterministic output, call the
network directly with `mc=False`, not through the generic bundle inferer
as-is.
"""


def _citations_section() -> str:
    return """## Citations

- McClure P. et al., "Knowing What You Know in Brain Segmentation Using
  Bayesian Deep Neural Networks", Front. Neuroinform. 2019 (the original
  kwyk architecture and published TensorFlow weights).
- TF -> PyTorch conversion: `nobrainer.datasets.convert_kwyk`. See
  `configs/metadata.json`'s `references` field and this checkpoint's
  originating `.provenance.json` sidecar for the exact source checkpoint
  path and SHA256.
"""


def main() -> None:
    args = parse_args()
    readme_path = Path(args.bundle_dir) / "docs" / "README.md"
    if not readme_path.exists():
        raise SystemExit(f"No docs/README.md found under {args.bundle_dir}")

    addendum = (
        "\n"
        + _citations_section()
        + "\n"
        + _DETERMINISM_CAVEAT
        + "\n"
        + _parity_section(args.parity_report, args.parity_stage)
    )
    with open(readme_path, "a") as f:
        f.write(addendum)
    print(f"Appended provenance/caveats to {readme_path}")


if __name__ == "__main__":
    main()
