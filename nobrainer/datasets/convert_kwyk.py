"""Convert pretrained kwyk TensorFlow SavedModel weights into KWYKMeshNet (PyTorch).

The kwyk architecture is already reimplemented in PyTorch
(``nobrainer/models/bayesian/kwyk_meshnet.py``). This module fills the remaining
gap: importing the *pretrained* TF SavedModel variables into that PyTorch model,
so users get the trusted published weights rather than a from-scratch retrain.

Primary input is a pre-extracted ``.npz`` (no TensorFlow dependency). A direct
SavedModel/checkpoint path is supported only if ``tensorflow`` is importable.

Every mapping claim below is verified against the actual
``neuronets/kwyk:latest-cpu`` container (SavedModel variables read via
``tf.train.NewCheckpointReader``) and validated numerically against the live TF
graph: the converted MAP model reproduces the original's logits to
``max|diff| ~ 1e-4`` (relative error ~4e-6) on identical input. Full evidence:
``docs/kwyk_mapping_verification.md``.

- TF conv filters are ``[k, k, k, in, out]`` -> torch ``[out, in, k, k, k]``
  via ``transpose(4, 3, 0, 1, 2)``.
- ``g`` is 5-D ``[1, 1, 1, 1, out]`` in every published checkpoint -> torch
  ``(out, 1, 1, 1, 1)``. A 1-D ``[out]`` form is also accepted (logged when
  taken -- it never occurs in real checkpoints).
- ``kernel_a`` / ``bias_a`` are raw params; forward uses ``abs(...)`` as sigma,
  so copy directly (no log/exp).
- concrete-dropout ``p`` is stored as a **probability** (all 672 published
  values lie in [0.608, 0.955]) -> ``p_logit = log(p / (1 - p))``, clamped to
  the model's forward clamp range first (see ``_CONCRETE_P_MIN/MAX``).
- ``bias_m``/``bias_a`` are present for **every** conv layer in **every**
  published model (24/24 across the three variants), so biases are imported
  by default; ``--drop-bias`` is an explicit opt-out.
- The output layer lives in its own ``logits/conv3d/*`` namespace (not
  ``layer_8/``) and is a full VWN conv; it maps 1:1 onto the model's FFG
  ``classifier`` with no information loss.
- ``ConcreteDropout3d`` stores its learnable logit as ``p_logit``, reached via
  ``layer_{i}`` -> ``dropout`` -> ``p_logit``. That key exists only when the
  model is built with ``dropout_type="concrete"``; the ``"bernoulli"`` variant
  uses a parameter-free ``nn.Dropout3d``, so emitting ``p_logit`` against it
  fails the strict load as an unexpected key.
- The layer-index base is not assumed -- it is detected and validated at
  runtime by :func:`detect_layer_index_base`.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# --- Constants (no magic numbers in logic) -------------------------------------

# TF [k,k,k,in,out] -> torch [out,in,k,k,k]
_CONV_PERM: tuple[int, ...] = (4, 3, 0, 1, 2)

# Concrete-dropout p is clamped to this range on every read of
# ConcreteDropout3d.p; match it exactly so the recovered p_logit reproduces
# forward behaviour. MUST stay in sync with vwn_layers.CONCRETE_P_MIN/MAX
# (kept as local literals so this module stays importable without nobrainer;
# a unit test asserts the two pairs are equal). Widened from [0.05, 0.95]
# because the published SSD checkpoint stores p up to 0.954055 -- the old
# ceiling silently clipped 598 of 672 values (89%). See
# docs/kwyk_mapping_verification.md, discrepancy D3.
_CONCRETE_P_MIN: float = 0.01
_CONCRETE_P_MAX: float = 0.99

# F.normalize uses eps=1e-12 internally; reproduce it in the numpy parity path.
_NORMALIZE_EPS: float = 1e-12

# Default parity tolerances (overridable via CLI).
_DEFAULT_ATOL: float = 1e-5
_DEFAULT_RTOL: float = 1e-6

# TF variable name templates. ``{i}`` is the layer index (base auto-detected).
_TF_CONV_V = "layer_{i}/conv3d/v"
_TF_CONV_G = "layer_{i}/conv3d/g"
_TF_CONV_KERNEL_A = "layer_{i}/conv3d/kernel_a"
_TF_CONV_BIAS_M = "layer_{i}/conv3d/bias_m"
_TF_CONV_BIAS_A = "layer_{i}/conv3d/bias_a"
_TF_CONCRETE_P = "layer_{i}/concrete_dropout/p"

# The output layer lives in its OWN namespace, ``logits/``, and is itself a
# VWN conv with bias and no concrete_dropout sibling. Verified against the
# actual ``neuronets/kwyk:latest-cpu`` container (all_50_bvwn_multi_prior,
# timestamp 1556816070): ``logits/conv3d/v [1,1,1,96,50]``, ``g``,
# ``kernel_a``, ``bias_m``, ``bias_a`` -- 48 variables total including
# ``global_step``, matching ARCHITECTURE.md's counts exactly. All five map
# 1:1 onto the FFG ``classifier`` (kwyk_meshnet.py), so nothing is discarded.
_TF_LOGITS_V = "logits/conv3d/v"
_TF_LOGITS_G = "logits/conv3d/g"
_TF_LOGITS_KERNEL_A = "logits/conv3d/kernel_a"
_TF_LOGITS_BIAS_M = "logits/conv3d/bias_m"
_TF_LOGITS_BIAS_A = "logits/conv3d/bias_a"


class ConversionError(RuntimeError):
    """Raised when the TF->PyTorch mapping cannot be completed safely."""


# --- TF variable loading -------------------------------------------------------


def load_tf_variables(npz_path: Path) -> dict[str, np.ndarray]:
    """Load TF variables from a pre-extracted ``.npz``.

    Keys must be exact TF variable names without the trailing ``:0`` (e.g.
    ``layer_1/conv3d/v``). This is the preferred path; it needs no TensorFlow.

    Parameters
    ----------
    npz_path : Path
        Path to the ``.npz`` produced by the extraction recipe (kwyk issue #15).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from TF variable name to array (float32-cast on read).
    """
    with np.load(npz_path) as data:
        return {key: np.asarray(data[key], dtype=np.float32) for key in data.files}


def load_tf_variables_from_savedmodel(tf_path: Path) -> dict[str, np.ndarray]:
    """Load TF variables directly from a SavedModel/checkpoint.

    Requires ``tensorflow``. If it is not importable, raise with a message
    pointing the user to the ``--npz`` path instead.
    """
    try:
        import tensorflow as tf  # noqa: PLC0415  (optional dependency by design)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ConversionError(
            "tensorflow is not installed; cannot read a SavedModel directly. "
            "Extract variables to an .npz (see kwyk issue #15) and pass --npz."
        ) from exc

    reader = tf.train.load_checkpoint(str(tf_path))
    shapes = reader.get_variable_to_shape_map()
    out: dict[str, np.ndarray] = {}
    for name in shapes:
        clean = name[:-2] if name.endswith(":0") else name
        out[clean] = np.asarray(reader.get_tensor(name), dtype=np.float32)
    return out


# --- Index-base detection ------------------------------------------------------


def detect_layer_index_base(tf_vars: dict[str, np.ndarray], n_layers: int) -> int:
    """Detect whether TF variable names are 0-based or 1-based, and validate.

    The PyTorch model uses ``layer_0 .. layer_{n_layers-1}``. The TF SavedModel
    documented in ARCHITECTURE.md uses ``layer_1 ..``. An off-by-one here loads
    every layer's weights into the wrong layer without necessarily erroring, so
    this is asserted explicitly rather than assumed.

    Returns
    -------
    int
        The detected base (0 or 1).
    """
    layer_indices: set[int] = set()
    pattern = re.compile(r"^layer_(\d+)/")
    for key in tf_vars:
        match = pattern.match(key)
        if match:
            layer_indices.add(int(match.group(1)))

    if not layer_indices:
        raise ConversionError(
            "No 'layer_<i>/...' variables found in the TF checkpoint. "
            "Check the .npz keys use exact TF names without ':0'."
        )

    lo, hi = min(layer_indices), max(layer_indices)
    span = hi - lo + 1
    # The output layer is NOT in this namespace (it is ``logits/conv3d/...``,
    # verified against the real container), so the layer_{i} span must equal
    # the hidden-layer count exactly.
    if span != n_layers:
        raise ConversionError(
            f"TF checkpoint spans {span} layers (indices {lo}..{hi}) but the "
            f"PyTorch model has {n_layers}. Refusing to convert on a mismatch."
        )
    if lo not in (0, 1):
        raise ConversionError(f"Unexpected TF layer base index {lo}; expected 0 or 1.")
    return lo


# --- Shape conversion helpers --------------------------------------------------


def _conv_tf_to_torch(arr: np.ndarray) -> np.ndarray:
    """TF conv filter ``[k,k,k,in,out]`` -> torch ``[out,in,k,k,k]``."""
    if arr.ndim != 5:
        raise ConversionError(
            f"Expected 5-D conv filter [k,k,k,in,out], got shape {arr.shape}."
        )
    return np.ascontiguousarray(np.transpose(arr, _CONV_PERM), dtype=np.float32)


def _g_tf_to_torch(arr: np.ndarray, out_channels: int) -> np.ndarray:
    """TF ``g`` (``[1,1,1,1,out]`` or ``[out]``) -> torch ``(out,1,1,1,1)``.

    ``np.transpose`` on a 1-D array is a no-op, so the two cases are branched
    explicitly on ``ndim`` rather than transposed blindly.
    """
    if arr.ndim == 1:
        if arr.size != out_channels:
            raise ConversionError(
                f"g has {arr.size} elements but out_channels={out_channels}."
            )
        # Every published kwyk checkpoint stores g as 5-D [1,1,1,1,out]; a
        # 1-D g is unusual enough to flag rather than silently reshape
        # (docs/kwyk_mapping_verification.md, discrepancy D4).
        logger.info(
            "g is 1-D (%d elements); real kwyk checkpoints store 5-D g -- "
            "reshaping, but verify the source checkpoint.",
            arr.size,
        )
        g = arr.reshape(out_channels, 1, 1, 1, 1)
    elif arr.ndim == 5:
        g = np.transpose(arr, _CONV_PERM)
    else:
        raise ConversionError(f"Unexpected g shape {arr.shape}; expected [out] or 5-D.")
    if g.shape != (out_channels, 1, 1, 1, 1):
        raise ConversionError(
            f"Converted g shape {g.shape} != expected {(out_channels, 1, 1, 1, 1)}."
        )
    return np.ascontiguousarray(g, dtype=np.float32)


def _p_to_logit(p: np.ndarray, name: str = "concrete_dropout/p") -> np.ndarray:
    """Concrete-dropout probability -> logit, clamped to the forward clamp range.

    Any value outside ``[_CONCRETE_P_MIN, _CONCRETE_P_MAX]`` is clipped and
    the count is logged as a warning -- clipping is a one-signed distortion,
    never silent (docs/kwyk_mapping_verification.md, discrepancy D3). With the
    current [0.01, 0.99] range, no published kwyk value is clipped.
    """
    p32 = p.astype(np.float32)
    n_clipped = int(((p32 < _CONCRETE_P_MIN) | (p32 > _CONCRETE_P_MAX)).sum())
    if n_clipped:
        logger.warning(
            "%s: %d of %d values fall outside [%.2f, %.2f] and were clipped "
            "(source range [%.6f, %.6f]) -- the converted dropout rates are "
            "distorted at these channels.",
            name,
            n_clipped,
            p32.size,
            _CONCRETE_P_MIN,
            _CONCRETE_P_MAX,
            float(p32.min()),
            float(p32.max()),
        )
    p_clamped = np.clip(p32, _CONCRETE_P_MIN, _CONCRETE_P_MAX)
    return np.log(p_clamped / (1.0 - p_clamped)).astype(np.float32)


# --- State-dict construction ---------------------------------------------------


def build_state_dict(
    model: torch.nn.Module,
    tf_vars: dict[str, np.ndarray],
    *,
    create_bias: bool,
) -> dict[str, torch.Tensor]:
    """Build a PyTorch state_dict from TF variables, matching model parameters.

    Parameters
    ----------
    model : torch.nn.Module
        A freshly constructed ``KWYKMeshNet`` (defines target shapes and the
        layer count via ``_n_layers``).
    tf_vars : dict[str, np.ndarray]
        TF variables keyed by exact name (no ``:0``).
    create_bias : bool
        If True (the default in :func:`convert` -- every published kwyk
        checkpoint carries biases for every conv layer), emit
        ``conv.bias_m`` / ``conv.bias_a`` entries from the TF biases. The
        model must have been built with bias-capable conv layers
        (``bias=True``); registering the parameters is sufficient because
        FFGConv3d.forward already consumes bias_m/bias_a. Setting this False
        (CLI: ``--drop-bias``) discards data measured to change the output
        logits by max ~20 on the published MAP weights.

    Returns
    -------
    dict[str, torch.Tensor]
        A state_dict intended for ``load_state_dict(..., strict=True)``.
    """
    n_layers = int(getattr(model, "_n_layers"))
    base = detect_layer_index_base(tf_vars, n_layers)
    state: dict[str, torch.Tensor] = {}

    for torch_i in range(n_layers):
        tf_i = torch_i + base
        conv_prefix = f"layer_{torch_i}.conv"

        v = tf_vars[_TF_CONV_V.format(i=tf_i)]
        g = tf_vars[_TF_CONV_G.format(i=tf_i)]
        kernel_a = tf_vars[_TF_CONV_KERNEL_A.format(i=tf_i)]

        v_t = _conv_tf_to_torch(v)
        out_channels = v_t.shape[0]
        state[f"{conv_prefix}.v"] = torch.from_numpy(v_t)
        state[f"{conv_prefix}.g"] = torch.from_numpy(_g_tf_to_torch(g, out_channels))
        state[f"{conv_prefix}.kernel_a"] = torch.from_numpy(_conv_tf_to_torch(kernel_a))

        bias_m_key = _TF_CONV_BIAS_M.format(i=tf_i)
        bias_a_key = _TF_CONV_BIAS_A.format(i=tf_i)
        has_bias = bias_m_key in tf_vars and bias_a_key in tf_vars
        if create_bias:
            if not has_bias:
                raise ConversionError(
                    f"Bias import requested but layer {tf_i} lacks "
                    "bias_m/bias_a in the TF checkpoint."
                )
            state[f"{conv_prefix}.bias_m"] = torch.from_numpy(
                np.ascontiguousarray(tf_vars[bias_m_key], dtype=np.float32)
            )
            state[f"{conv_prefix}.bias_a"] = torch.from_numpy(
                np.ascontiguousarray(tf_vars[bias_a_key], dtype=np.float32)
            )
        elif has_bias:
            logger.warning(
                "Layer %d has TF bias_m/bias_a but bias import is disabled "
                "(--drop-bias): DROPPING them. Measured on the published MAP "
                "weights this shifts output logits by max ~20 (vs ~1e-4 with "
                "biases kept) -- the result is NOT faithful to the original.",
                tf_i,
            )

        # Confirmed: ConcreteDropout3d stores its learnable logit as
        # ``p_logit`` (vwn_layers.py:197), so ``layer_{i}.dropout.p_logit`` is
        # the correct key -- but only for a model built with
        # dropout_type="concrete". Against the default "bernoulli" variant
        # (parameter-free nn.Dropout3d) the strict load rejects it, which is
        # the intended loud failure rather than a silent mismatch.
        p_key = _TF_CONCRETE_P.format(i=tf_i)
        if p_key in tf_vars:
            state[f"layer_{torch_i}.dropout.p_logit"] = torch.from_numpy(
                _p_to_logit(tf_vars[p_key], name=p_key)
            )

    _add_classifier(state, tf_vars)
    return state


def _add_classifier(
    state: dict[str, torch.Tensor],
    tf_vars: dict[str, np.ndarray],
) -> None:
    """Map the TF VWN ``logits/`` layer 1:1 onto the FFG ``classifier``.

    The TF output layer (``logits/conv3d/...``, its own namespace -- verified
    against the real container) is a full VWN conv, and
    ``KWYKMeshNet.classifier`` is an ``FFGConv3d`` as well, so all five
    variables map directly with **no information loss** -- including the
    ``kernel_a``/``bias_a`` sigmas that a plain-conv classifier could not
    represent and whose absence made MC uncertainty under-dispersed at the
    output (docs/kwyk_mapping_verification.md, discrepancy D2).
    """
    required = (
        _TF_LOGITS_V,
        _TF_LOGITS_G,
        _TF_LOGITS_KERNEL_A,
        _TF_LOGITS_BIAS_M,
        _TF_LOGITS_BIAS_A,
    )
    missing = [k for k in required if k not in tf_vars]
    if missing:
        raise ConversionError(
            f"Checkpoint has no logits layer (missing {missing}). The model's "
            "classifier cannot be filled, and a partial state_dict would fail "
            "the strict load anyway."
        )

    v_t = _conv_tf_to_torch(tf_vars[_TF_LOGITS_V])
    out_channels = v_t.shape[0]
    state["classifier.v"] = torch.from_numpy(v_t)
    state["classifier.g"] = torch.from_numpy(
        _g_tf_to_torch(tf_vars[_TF_LOGITS_G], out_channels)
    )
    state["classifier.kernel_a"] = torch.from_numpy(
        _conv_tf_to_torch(tf_vars[_TF_LOGITS_KERNEL_A])
    )
    state["classifier.bias_m"] = torch.from_numpy(
        np.ascontiguousarray(tf_vars[_TF_LOGITS_BIAS_M], dtype=np.float32)
    )
    state["classifier.bias_a"] = torch.from_numpy(
        np.ascontiguousarray(tf_vars[_TF_LOGITS_BIAS_A], dtype=np.float32)
    )


# --- Parity check --------------------------------------------------------------


def _kernel_m_numpy(v_torch: np.ndarray, g_torch: np.ndarray) -> np.ndarray:
    """Reproduce ``FFGConv3d.kernel_m`` = ``g * normalize(v.flatten(1))`` in numpy.

    Mirrors ``F.normalize(self.v.flatten(1), dim=1).view_as(self.v)`` with the
    same L2 norm over dims (in, k, k, k) per output channel and eps=1e-12.
    """
    out = v_torch.shape[0]
    v_flat = v_torch.reshape(out, -1)
    norms = np.linalg.norm(v_flat, axis=1, keepdims=True)
    norms = np.maximum(norms, _NORMALIZE_EPS)
    v_norm = (v_flat / norms).reshape(v_torch.shape)
    return (g_torch * v_norm).astype(np.float32)


def offline_parity_check(
    model: torch.nn.Module,
    tf_vars: dict[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
    seed: int = 0,
) -> None:
    """Verify parameter mapping without a TF runtime, layer by layer.

    For each conv layer, recompute ``kernel_m`` from the TF ``v``/``g`` in numpy,
    run a deterministic ``F.conv3d`` mean path, and compare against the model's
    own ``kernel_m`` conv on the same random input. This isolates the mapping
    (axis order, normalization, bias) from every other confound.

    Scope: this validates the mapping *transform* (axis permutation, weight-norm,
    bias arithmetic), not source-data integrity. Corruption identical on both the
    recompute and the load path would move together and pass. For end-to-end
    validation against the original model outputs, run the full-graph comparison
    in ``scripts/kwyk_reproduction/05_compare_kwyk.py`` (Stage 3).

    Raises
    ------
    ConversionError
        If any layer's mean-path output diverges beyond tolerance.
    """
    torch.manual_seed(seed)
    n_layers = int(getattr(model, "_n_layers"))
    base = detect_layer_index_base(tf_vars, n_layers)
    model.eval()

    # Hidden convs plus the FFG classifier -- the output layer is part of the
    # mapping and must be part of the check.
    entries: list[tuple[str, torch.nn.Module, str, str]] = [
        (
            f"layer_{torch_i} (TF layer_{torch_i + base})",
            getattr(model, f"layer_{torch_i}").conv,
            _TF_CONV_V.format(i=torch_i + base),
            _TF_CONV_G.format(i=torch_i + base),
        )
        for torch_i in range(n_layers)
    ]
    entries.append(
        ("classifier (TF logits)", model.classifier, _TF_LOGITS_V, _TF_LOGITS_G)
    )

    with torch.no_grad():
        for name, conv, v_key, g_key in entries:
            v_t = _conv_tf_to_torch(tf_vars[v_key])
            g_t = _g_tf_to_torch(tf_vars[g_key], v_t.shape[0])
            kernel_m_ref = torch.from_numpy(_kernel_m_numpy(v_t, g_t))

            in_ch = v_t.shape[1]
            x = torch.randn(1, in_ch, 8, 8, 8)

            bias_m = conv.bias_m if conv.bias_m is not None else None
            out_ref = F.conv3d(
                x, kernel_m_ref, bias_m, conv.stride, conv.padding, conv.dilation
            )
            out_model = F.conv3d(
                x, conv.kernel_m, bias_m, conv.stride, conv.padding, conv.dilation
            )

            if not torch.allclose(out_ref, out_model, atol=atol, rtol=rtol):
                max_abs = (out_ref - out_model).abs().max().item()
                raise ConversionError(
                    f"Parity failed at {name}: "
                    f"max_abs_diff={max_abs:.3e} exceeds atol={atol:.1e}. "
                    "Likely an axis-order, normalization, or bias mismatch."
                )
    logger.info(
        "Offline parity check passed for all %d conv layers (incl. classifier).",
        len(entries),
    )


# --- Orchestration -------------------------------------------------------------


def convert(
    model: torch.nn.Module,
    *,
    npz: Path | None = None,
    tf_path: Path | None = None,
    create_bias: bool = True,
    run_parity: bool = True,
    atol: float = _DEFAULT_ATOL,
    rtol: float = _DEFAULT_RTOL,
) -> torch.nn.Module:
    """Load TF variables, build a strict state_dict, verify parity, and load it.

    Exactly one of ``npz`` or ``tf_path`` must be given.

    ``create_bias`` defaults to True: every published kwyk checkpoint carries
    ``bias_m``/``bias_a`` for every conv layer, and dropping them changes the
    output logits by max ~20 (docs/kwyk_mapping_verification.md, D1). Pass
    False only deliberately, with a model built ``bias=False``.
    """
    if (npz is None) == (tf_path is None):
        raise ConversionError("Provide exactly one of --npz or --tf-path.")

    tf_vars = (
        load_tf_variables(npz) if npz else load_tf_variables_from_savedmodel(tf_path)
    )

    state = build_state_dict(model, tf_vars, create_bias=create_bias)
    # strict=True is a free correctness gate: any missing/unexpected key
    # (a name typo, a dropped bias, a wrong layer count) fails loudly here.
    model.load_state_dict(state, strict=True)
    logger.info("Loaded converted weights into model (strict=True).")

    # Parity runs AFTER loading: it independently recomputes kernel_m from the
    # raw TF v/g and checks the loaded model reproduces it. Running before the
    # load would compare against random init and always fail.
    if run_parity:
        offline_parity_check(model, tf_vars, atol=atol, rtol=rtol)

    return model


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", type=Path, help="Pre-extracted TF variables (preferred).")
    src.add_argument(
        "--tf-path", type=Path, help="SavedModel/checkpoint (requires tensorflow)."
    )
    parser.add_argument(
        "--drop-bias",
        action="store_true",
        help=(
            "Discard the TF conv biases and build a bias-free model. Every "
            "published kwyk checkpoint carries bias_m/bias_a for every conv "
            "layer, and dropping them shifts output logits by max ~20 on the "
            "MAP weights -- so the DEFAULT is to keep them (model built with "
            "bias=True). This flag is an explicit, logged opt-out."
        ),
    )
    parser.add_argument(
        "--dropout-type",
        default="bernoulli",
        choices=("bernoulli", "concrete"),
        help=(
            "Must be 'concrete' to import a checkpoint containing "
            "concrete_dropout/p: only that variant has a learnable "
            "dropout.p_logit parameter (the bernoulli variant uses a "
            "parameter-free nn.Dropout3d, so the key would be unexpected)."
        ),
    )
    parser.add_argument("--n-classes", type=int, required=True)
    parser.add_argument(
        "--filters",
        type=int,
        default=96,
        help="Hidden-layer filter count. The published kwyk models use 96.",
    )
    parser.add_argument(
        "--receptive-field",
        type=int,
        default=37,
        choices=(37, 67, 129),
        help="Dilation schedule selector. The published kwyk models use 37.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output .pth path.")
    parser.add_argument("--no-parity", action="store_true", help="Skip parity check.")
    parser.add_argument("--atol", type=float, default=_DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=_DEFAULT_RTOL)
    return parser


def _sha256_of_file(path: Path) -> str | None:
    """SHA-256 hex digest of a regular file; None for a directory/missing path."""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_provenance(out: Path, args: argparse.Namespace, create_bias: bool) -> Path:
    """Write a ``<out stem>.provenance.json`` sidecar next to the weights.

    The MAP and BD checkpoints are structurally indistinguishable (identical
    key sets and shapes -- docs/kwyk_mapping_verification.md, D5), so the
    source path and its content hash are the only durable record of which
    model a ``.pth`` came from. The architecture args are recorded because
    the state_dict does not store them (dropout_type, receptive_field, etc.
    are constructor-time-only).
    """
    source = args.npz or args.tf_path
    provenance = {
        "source": str(source),
        "source_sha256": _sha256_of_file(source),
        "model_registry_name": "kwyk_meshnet",
        "n_classes": args.n_classes,
        "filters": args.filters,
        "receptive_field": args.receptive_field,
        "dropout_type": args.dropout_type,
        "bias": create_bias,
        "converter": "nobrainer.datasets.convert_kwyk",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    prov_path = out.with_name(out.stem + ".provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return prov_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    create_bias = not args.drop_bias

    # Imported here so the module's helpers are usable without nobrainer present
    # (e.g. in unit tests that construct a stub model).
    from nobrainer.models import get as get_model  # noqa: PLC0415

    # The registry name is "kwyk_meshnet" (nobrainer/models/__init__.py); there
    # is no "kwyk" export. bias must mirror create_bias, because a strict
    # state_dict load rejects conv.bias_m/bias_a against a bias-free model
    # (and vice versa reports them missing).
    model = get_model("kwyk_meshnet")(
        n_classes=args.n_classes,
        filters=args.filters,
        receptive_field=args.receptive_field,
        dropout_type=args.dropout_type,
        bias=create_bias,
    )
    convert(
        model,
        npz=args.npz,
        tf_path=args.tf_path,
        create_bias=create_bias,
        run_parity=not args.no_parity,
        atol=args.atol,
        rtol=args.rtol,
    )
    torch.save(model.state_dict(), args.out)
    logger.info("Wrote converted weights to %s", args.out)
    prov_path = _write_provenance(args.out, args, create_bias)
    logger.info("Wrote provenance sidecar to %s", prov_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
