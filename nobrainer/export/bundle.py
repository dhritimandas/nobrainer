"""Export a trained nobrainer estimator as a MONAI model-zoo bundle.

See ``https://docs.monai.io/en/stable/mb_specification.html`` for the bundle
directory layout and ``configs/metadata.json`` schema this module targets.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import warnings

import monai
import numpy as np
import torch

import nobrainer
from nobrainer.models import get as get_model

MONAI_META_SCHEMA_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/"
    "releases/download/0.8.1/meta_schema_20240725.json"
)

# Fully-convolutional / segmentation-contract architectures this exporter
# understands.  Excludes autoencoder, simsiam (multi-input forward),
# dcgan/progressivegan (Lightning modules with no single-tensor forward) --
# none of these describe a (B, n_classes, D, H, W) segmentation contract.
SUPPORTED_ARCHITECTURES = frozenset(
    {
        "unet",
        "vnet",
        "attention_unet",
        "unetr",
        "meshnet",
        "highresnet",
        "swin_unetr",
        "segresnet",
        "segformer3d",
        "bayesian_meshnet",
        "bayesian_vnet",
        "kwyk_meshnet",
    }
)

NOT_FOR_CLINICAL_USE = (
    "**NOT FOR CLINICAL USE.** This model is exported for research purposes "
    "only. It has not been evaluated, reviewed, or approved for clinical "
    "diagnosis, treatment planning, or any other clinical use."
)


class BundleExportError(RuntimeError):
    """Raised when a trained model cannot be exported as a MONAI bundle."""


def _package_version(name: str) -> str:
    """Return the installed version of a distribution.

    Parameters
    ----------
    name : str
        Distribution name as registered with ``importlib.metadata``.

    Returns
    -------
    str
        Installed version string, read live -- never hardcoded.
    """
    return importlib.metadata.version(name)


def _extra_required_packages(base_model: str) -> dict[str, str]:
    """Return extra required packages (beyond nobrainer/nibabel) for an architecture.

    Parameters
    ----------
    base_model : str
        Registry name of the exported architecture.

    Returns
    -------
    dict of str to str
        Package name mapped to installed version.
    """
    extra: dict[str, str] = {}
    if base_model == "segformer3d":
        extra["einops"] = _package_version("einops")
    if base_model in {"bayesian_meshnet", "bayesian_vnet", "kwyk_meshnet"}:
        try:
            extra["pyro-ppl"] = _package_version("pyro-ppl")
        except importlib.metadata.PackageNotFoundError:
            pass
    return extra


def _resolve_in_channels(base_model: str, model_args: dict[str, Any]) -> int:
    """Resolve ``in_channels`` for an architecture.

    Croissant provenance does not currently record ``in_channels``
    (see ``nobrainer/processing/croissant.py``), so it is read from
    ``model_args`` if present, else from the factory's declared default.

    Parameters
    ----------
    base_model : str
        Registry name of the architecture.
    model_args : dict
        Stored ``model_args`` from the estimator's provenance.

    Returns
    -------
    int
        Number of input channels.

    Raises
    ------
    BundleExportError
        If ``in_channels`` cannot be resolved.
    """
    if "in_channels" in model_args:
        return int(model_args["in_channels"])
    factory = get_model(base_model)
    default = inspect.signature(factory).parameters["in_channels"].default
    if default is inspect.Parameter.empty:
        raise BundleExportError(
            f"Cannot resolve in_channels for '{base_model}': not present in "
            "model_args and the factory has no default."
        )
    return int(default)


def _resolve_spatial_shape(
    base_model: str,
    model_args: dict[str, Any],
    block_shape: tuple[int, ...],
    spatial_shape_override: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    """Resolve the 3-D spatial patch shape used for the bundle contract.

    Uses the literal ``block_shape`` recorded at training time rather than a
    symbolic divisibility expression. MONAI's ``spatial_shape`` grammar in
    ``metadata.json`` only binds the variables ``p`` and ``n``
    (``monai.bundle.scripts._get_fake_spatial_shape``), which cannot express
    "multiple of 32 and >= 64" -- the actual constraint measured for
    ``swin_unetr``. ``block_shape`` is also exactly the value used for the
    bundle's ``SlidingWindowInferer.roi_size``, so the two cannot drift.

    Parameters
    ----------
    base_model : str
        Registry name of the architecture.
    model_args : dict
        Stored ``model_args`` from provenance.
    block_shape : tuple of int
        ``block_shape`` recorded in the estimator's provenance.
    spatial_shape_override : tuple of int, or None
        User-supplied override, if any.

    Returns
    -------
    tuple of int
        Three-element spatial shape.

    Raises
    ------
    BundleExportError
        If no 3-D shape can be resolved, or (for ``unetr``) the resolved
        shape does not match the ``img_size`` baked into the weights.
    """
    if spatial_shape_override is not None:
        shape = tuple(int(v) for v in spatial_shape_override)
    elif len(block_shape) == 3:
        shape = tuple(int(v) for v in block_shape)
    else:
        raise BundleExportError(
            "block_shape in the model's provenance is missing or not 3-D "
            f"(got {block_shape!r}). Pass an explicit spatial_shape override."
        )

    if base_model == "unetr":
        img_size = model_args.get("img_size")
        if img_size is not None and tuple(int(v) for v in img_size) != shape:
            raise BundleExportError(
                f"unetr weights were trained with img_size={tuple(img_size)}, "
                f"which does not match the resolved spatial_shape {shape}. "
                "UNETR bakes img_size into its weights; the two must match."
            )
    return shape


def build_metadata(
    *,
    base_model: str,
    in_channels: int,
    n_classes: int,
    spatial_shape: tuple[int, int, int],
    provenance: dict[str, Any],
    version: str,
    name: str | None,
    task: str | None,
    description: str | None,
    authors: str,
    copyright_: str,
    labels: list[str] | None,
    references: list[str] | None,
    stochastic: bool,
) -> dict[str, Any]:
    """Build a MONAI bundle ``configs/metadata.json`` dict.

    Emits all 11 schema-required top-level keys plus the optional keys the
    MONAI model-zoo bundles ship. Version fields are read from the live
    installed packages, never hardcoded.

    Parameters
    ----------
    base_model : str
        Registry name of the exported architecture.
    in_channels : int
        Number of input channels.
    n_classes : int
        Number of output classes.
    spatial_shape : tuple of int
        Expected input/output spatial patch shape.
    provenance : dict
        The ``nobrainer:provenance`` block from ``croissant.json``.
    version : str
        Bundle version string.
    name : str or None
        Human-readable bundle name; a default is generated if None.
    task : str or None
        Task description; a default is generated if None.
    description : str or None
        Longer description; a default is generated if None.
    authors : str
        Author string.
    copyright_ : str
        Copyright string.
    labels : list of str, or None
        Class label names, index-ordered starting at background=0. Must have
        length ``n_classes`` if given.
    references : list of str, or None
        Reference citations.
    stochastic : bool
        Whether the network was found to be non-deterministic across two
        identical forward passes (Bayesian/MC models).

    Returns
    -------
    dict
        The ``metadata.json`` content.

    Raises
    ------
    BundleExportError
        If ``labels`` is given but its length does not match ``n_classes``.
    """
    pytorch_version = torch.__version__.split("+")[0]
    required_packages = {
        "nobrainer": nobrainer.__version__,
        "nibabel": _package_version("nibabel"),
        **_extra_required_packages(base_model),
    }

    if labels is None:
        channel_def = {"0": "background"}
        channel_def.update({str(i): f"class_{i}" for i in range(1, n_classes)})
    else:
        if len(labels) != n_classes:
            raise BundleExportError(
                f"labels has {len(labels)} entries but n_classes={n_classes}."
            )
        channel_def = {str(i): label for i, label in enumerate(labels)}

    intended_use = (
        "Research use only; not a substitute for expert diagnosis. "
        + NOT_FOR_CLINICAL_USE
    )
    if stochastic:
        intended_use += (
            " This network is stochastic (Bayesian): each forward pass "
            "returns one posterior draw, not a deterministic prediction."
        )

    best_loss = provenance.get("best_loss")
    data_source = ", ".join(
        d.get("path", "") for d in provenance.get("source_datasets", []) if d
    )

    return {
        "schema": MONAI_META_SCHEMA_URL,
        "version": version,
        "changelog": {version: f"Exported from nobrainer {nobrainer.__version__}"},
        "monai_version": monai.__version__,
        "pytorch_version": pytorch_version,
        "numpy_version": np.__version__,
        "required_packages_version": required_packages,
        "name": name or f"Nobrainer {base_model} segmentation",
        "task": task or "3D brain MRI segmentation",
        "description": description
        or (
            f"3-D brain MRI segmentation ({base_model}) exported from "
            f"nobrainer {nobrainer.__version__}."
        ),
        "authors": authors,
        "copyright": copyright_,
        "data_source": data_source,
        "data_type": "nibabel",
        "image_classes": f"{in_channels}-channel MRI, intensity scaled to [0, 1]",
        "label_classes": f"{n_classes} classes, one-hot",
        "pred_classes": f"{n_classes} channels OneHot data",
        "eval_metrics": {"best_loss": best_loss} if best_loss is not None else {},
        "intended_use": intended_use,
        "references": references or [],
        "network_data_format": {
            "inputs": {
                "image": {
                    "type": "image",
                    "format": "magnitude",
                    "modality": "MR",
                    "num_channels": in_channels,
                    "spatial_shape": list(spatial_shape),
                    "dtype": "float32",
                    "value_range": [0, 1],
                    "is_patch_data": True,
                    "channel_def": {"0": "image"},
                }
            },
            "outputs": {
                "pred": {
                    "type": "image",
                    # The MONAI spec reserves "labels" for N one-hot channels
                    # and "segmentation" for single-channel categorical
                    # output, but both real model-zoo bundles (spleen_ct,
                    # wholeBrainSeg) use "segmentation" for one-hot output.
                    # Matched here since downstream consumers target the zoo.
                    "format": "segmentation",
                    "num_channels": n_classes,
                    "spatial_shape": list(spatial_shape),
                    "dtype": "float32",
                    "value_range": [0, 1],
                    "is_patch_data": True,
                    "channel_def": channel_def,
                }
            },
        },
    }


def build_inference_config(
    *,
    base_model: str,
    model_args: dict[str, Any],
    in_channels: int,
    n_classes: int,
    spatial_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Build a runnable MONAI bundle ``configs/inference.json`` dict.

    ``network_def._target_`` is a fully-qualified dotted path into
    ``nobrainer.models`` (nobrainer factories are not in MONAI's
    ``ComponentLocator`` namespace); ``monai.bundle.ConfigParser`` resolves
    dotted paths via ``pydoc.locate`` (verified at plan time).

    Parameters
    ----------
    base_model : str
        Registry name of the architecture.
    model_args : dict
        Stored ``model_args`` from provenance (channels, strides, etc.).
    in_channels : int
        Number of input channels.
    n_classes : int
        Number of output classes.
    spatial_shape : tuple of int
        Expected spatial patch shape; used as the sliding-window ROI size.

    Returns
    -------
    dict
        The ``inference.json`` content.
    """
    factory = get_model(base_model)
    target = f"{factory.__module__}.{factory.__name__}"

    network_kwargs = {
        k: v for k, v in model_args.items() if k not in ("n_classes", "in_channels")
    }
    network_kwargs["n_classes"] = n_classes
    network_kwargs["in_channels"] = in_channels

    return {
        "imports": ["$import glob"],
        "device": "$torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')",
        "ckpt_path": "$@bundle_root + '/models/model.pt'",
        "dataset_dir": "/workspace/data",
        "datalist": "$list(sorted(glob.glob(@dataset_dir + '/*.nii.gz')))",
        "network_def": {"_target_": target, **network_kwargs},
        "network": "$@network_def.to(@device)",
        "preprocessing": {
            "_target_": "Compose",
            "transforms": [
                {"_target_": "LoadImaged", "keys": "image"},
                {"_target_": "EnsureChannelFirstd", "keys": "image"},
                {"_target_": "ScaleIntensityd", "keys": "image"},
                {"_target_": "EnsureTyped", "keys": "image", "device": "@device"},
            ],
        },
        "dataset": {
            "_target_": "Dataset",
            "data": "$[{'image': i} for i in @datalist]",
            "transform": "@preprocessing",
        },
        "dataloader": {
            "_target_": "DataLoader",
            "dataset": "@dataset",
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
        },
        "inferer": {
            "_target_": "SlidingWindowInferer",
            "roi_size": list(spatial_shape),
            "sw_batch_size": 1,
            "overlap": 0.25,
        },
        "postprocessing": {
            "_target_": "Compose",
            "transforms": [
                {"_target_": "Activationsd", "keys": "pred", "softmax": True},
                {"_target_": "AsDiscreted", "keys": "pred", "argmax": True},
                {
                    "_target_": "SaveImaged",
                    "keys": "pred",
                    "meta_keys": "image_meta_dict",
                    "output_dir": "$@bundle_root + '/eval'",
                },
            ],
        },
        "handlers": [
            {
                "_target_": "CheckpointLoader",
                "load_path": "@ckpt_path",
                "load_dict": {"model": "@network"},
            }
        ],
        "evaluator": {
            "_target_": "SupervisedEvaluator",
            "device": "@device",
            "val_data_loader": "@dataloader",
            "network": "@network",
            "inferer": "@inferer",
            "postprocessing": "@postprocessing",
            "val_handlers": "@handlers",
        },
        "evaluating": ["$@evaluator.run()"],
    }


def _write_license(output_dir: Path) -> None:
    """Write ``LICENSE`` into the bundle root.

    Copies nobrainer's own repository LICENSE file (Apache-2.0, per
    ``pyproject.toml``) when it can be located relative to the installed
    package; otherwise writes a short pointer to it.

    Parameters
    ----------
    output_dir : Path
        Bundle root directory.
    """
    src = Path(nobrainer.__file__).resolve().parent.parent / "LICENSE"
    dest = output_dir / "LICENSE"
    if src.exists():
        shutil.copyfile(src, dest)
    else:
        dest.write_text(
            "Apache License 2.0. See "
            "https://github.com/neuronets/nobrainer/blob/main/LICENSE\n"
        )


def _write_readme(
    docs_dir: Path,
    *,
    base_model: str,
    n_classes: int,
    in_channels: int,
    spatial_shape: tuple[int, int, int],
    stochastic: bool,
) -> None:
    """Write ``docs/README.md`` with run instructions and a use disclaimer.

    Parameters
    ----------
    docs_dir : Path
        Bundle ``docs/`` directory.
    base_model : str
        Registry name of the architecture.
    n_classes : int
        Number of output classes.
    in_channels : int
        Number of input channels.
    spatial_shape : tuple of int
        Expected spatial patch shape.
    stochastic : bool
        Whether the network is non-deterministic across identical inputs.
    """
    stochastic_note = (
        "\n**Note:** this network is stochastic (Bayesian); repeated runs "
        "on the same input yield different outputs.\n"
        if stochastic
        else ""
    )
    readme = f"""# Nobrainer {base_model} bundle

{NOT_FOR_CLINICAL_USE}
{stochastic_note}
Architecture: `{base_model}` ({in_channels} input channel(s), {n_classes} \
output classes).
Expected patch shape: {list(spatial_shape)}.

## Run inference

```
python -m monai.bundle run \\
    --meta_file configs/metadata.json \\
    --config_file configs/inference.json \\
    --dataset_dir ./input \\
    --bundle_root .
```
"""
    (docs_dir / "README.md").write_text(readme)


def _verify_metadata_subprocess(meta_file: Path) -> None:
    """Validate ``metadata.json`` against the MONAI bundle schema.

    Shells out to ``python -m monai.bundle verify_metadata`` (downloads the
    schema referenced by the ``schema`` key and validates against it) so the
    written bundle is checked by MONAI's own validator, not a reimplementation
    of it. Requires the optional ``nobrainer[bundle]`` extra (``fire``,
    ``jsonschema``) -- MONAI's own CLI entry point and ``verify_metadata``
    depend on them; nobrainer's core dependencies do not.

    The downloaded schema is cached at a stable path under the system temp
    directory so repeated calls (e.g. across a test session) do not
    re-download it.

    Parameters
    ----------
    meta_file : Path
        Path to the written ``configs/metadata.json``.

    Raises
    ------
    BundleExportError
        If ``verify_metadata`` reports an actual schema violation.
    """
    schema_cache = (
        Path(tempfile.gettempdir()) / "nobrainer_monai_bundle_meta_schema.json"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "monai.bundle",
            "verify_metadata",
            "--meta_file",
            str(meta_file),
            "--filepath",
            str(schema_cache),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    combined = result.stdout + result.stderr
    if "OptionalImportError" in combined or "ModuleNotFoundError" in combined:
        warnings.warn(
            "Skipped MONAI schema validation: `python -m monai.bundle "
            "verify_metadata` requires the optional 'fire' and 'jsonschema' "
            "packages (install with `uv pip install -e '.[bundle]'`). The "
            f"bundle was still written to disk.\n{combined}",
            stacklevel=2,
        )
        return

    raise BundleExportError(
        "monai.bundle verify_metadata rejected the exported bundle:\n"
        f"{result.stdout}\n{result.stderr}"
    )


def export_bundle(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    torchscript: bool = True,
    trace: bool = False,
    allow_stochastic: bool = False,
    spatial_shape: tuple[int, int, int] | None = None,
    version: str = "0.0.1",
    name: str | None = None,
    task: str | None = None,
    description: str | None = None,
    authors: str = "nobrainer contributors",
    copyright_: str = "Copyright (c) nobrainer contributors",
    labels: list[str] | None = None,
    references: list[str] | None = None,
    verify: bool = True,
) -> Path:
    """Export a saved nobrainer estimator directory as a MONAI bundle.

    Parameters
    ----------
    model_dir : str or Path
        Directory written by ``Segmentation.save()`` (``model.pth`` +
        ``croissant.json``).
    output_dir : str or Path
        Bundle directory to create. Must not already exist.
    torchscript : bool
        Attempt a TorchScript export (``models/model.ts``). On failure a
        warning is emitted and ``model.ts`` is omitted -- it is optional per
        the bundle spec.
    trace : bool
        Force ``torch.jit.trace`` instead of ``torch.jit.script``. Bakes in
        a fixed input shape; not the automatic fallback for a script
        failure.
    allow_stochastic : bool
        Required to export a network whose output is non-deterministic
        across two identical forward passes (Bayesian/MC models).
    spatial_shape : tuple of int, or None
        Override for the patch shape; defaults to the provenance
        ``block_shape``.
    version, name, task, description, authors, copyright_, labels, references
        Passed through to :func:`build_metadata`.
    verify : bool
        Run ``monai.bundle verify_metadata`` on the written bundle and raise
        on failure.

    Returns
    -------
    Path
        Path to the written bundle directory (``output_dir``).

    Raises
    ------
    BundleExportError
        If the output directory exists, the architecture is unsupported,
        the spatial shape cannot be resolved, the model is stochastic
        without ``allow_stochastic``, or ``verify_metadata`` rejects the
        result.
    """
    from monai.networks.utils import convert_to_torchscript, save_state

    from nobrainer.processing.segmentation import Segmentation

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise BundleExportError(f"Output directory already exists: {output_dir}")

    estimator = Segmentation.load(model_dir)
    base_model = estimator.base_model
    if base_model not in SUPPORTED_ARCHITECTURES:
        raise BundleExportError(
            f"Architecture '{base_model}' is not supported for bundle "
            f"export. Supported: {sorted(SUPPORTED_ARCHITECTURES)}."
        )

    n_classes = estimator.n_classes_
    if not n_classes:
        raise BundleExportError(
            "n_classes missing from the model's provenance; cannot export."
        )
    model_args = dict(estimator.model_args)
    in_channels = _resolve_in_channels(base_model, model_args)
    shape = _resolve_spatial_shape(
        base_model, model_args, tuple(estimator.block_shape_ or ()), spatial_shape
    )

    net = estimator.model_
    net.eval()
    probe = torch.rand(1, in_channels, *shape)
    with torch.no_grad():
        out_a = net(probe)
        out_b = net(probe)
    if tuple(out_a.shape) != (1, n_classes, *shape):
        raise BundleExportError(
            f"Self-verification failed: expected output shape "
            f"(1, {n_classes}, {tuple(shape)}), got {tuple(out_a.shape)}."
        )
    is_stochastic = not torch.allclose(out_a, out_b)
    if is_stochastic and not allow_stochastic:
        raise BundleExportError(
            f"'{base_model}' produced different output on two identical "
            "forward passes (stochastic/Bayesian network). Pass "
            "allow_stochastic=True to export it anyway; the resulting "
            "bundle documents that inference yields one posterior draw."
        )

    provenance = json.loads((model_dir / "croissant.json").read_text()).get(
        "nobrainer:provenance", {}
    )

    metadata = build_metadata(
        base_model=base_model,
        in_channels=in_channels,
        n_classes=n_classes,
        spatial_shape=shape,
        provenance=provenance,
        version=version,
        name=name,
        task=task,
        description=description,
        authors=authors,
        copyright_=copyright_,
        labels=labels,
        references=references,
        stochastic=is_stochastic,
    )
    inference_config = build_inference_config(
        base_model=base_model,
        model_args=model_args,
        in_channels=in_channels,
        n_classes=n_classes,
        spatial_shape=shape,
    )

    configs_dir = output_dir / "configs"
    models_dir = output_dir / "models"
    docs_dir = output_dir / "docs"
    configs_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)

    (configs_dir / "metadata.json").write_text(json.dumps(metadata, indent=4))
    (configs_dir / "inference.json").write_text(json.dumps(inference_config, indent=4))

    save_state(net, str(models_dir / "model.pt"))

    if torchscript:
        try:
            if trace:
                convert_to_torchscript(
                    model=net,
                    filename_or_obj=str(models_dir / "model.ts"),
                    inputs=[probe],
                    use_trace=True,
                )
            else:
                convert_to_torchscript(
                    model=net,
                    filename_or_obj=str(models_dir / "model.ts"),
                )
        except Exception as exc:  # noqa: BLE001 - any scripting/tracing failure
            warnings.warn(
                f"torch.jit.{'trace' if trace else 'script'} failed for "
                f"'{base_model}': {type(exc).__name__}: {exc}. Skipping "
                "models/model.ts; the bundle remains spec-valid (model.ts "
                "is optional).",
                stacklevel=2,
            )
            (models_dir / "model.ts").unlink(missing_ok=True)

    _write_license(output_dir)
    _write_readme(
        docs_dir,
        base_model=base_model,
        n_classes=n_classes,
        in_channels=in_channels,
        spatial_shape=shape,
        stochastic=is_stochastic,
    )

    if verify:
        _verify_metadata_subprocess(configs_dir / "metadata.json")

    return output_dir
