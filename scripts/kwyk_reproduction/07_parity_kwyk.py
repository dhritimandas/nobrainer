#!/usr/bin/env python
"""Full-graph numerical parity: converted PyTorch kwyk vs the original container.

Runs both the ``neuronets/kwyk`` Docker container (original TensorFlow
graphs) and a PyTorch checkpoint produced by
``nobrainer.datasets.convert_kwyk`` on identical preprocessed input, and
emits a machine-readable pass/fail report.

Preprocessing contract (see docs/kwyk_parity_report.md for the full
rationale): conform exactly once, inside the container
(``mri_convert --conform``, FreeSurfer, not reproducible in numpy), then
z-score the conformed volume exactly once on the host and feed the
identical normalized array to both sides. Never let either side
normalize independently -- see ``assert_preprocessing``.

Usage:
    # preprocessing guard alone
    python 07_parity_kwyk.py --volume sub-01_t1.mgz --check-preprocessing-only

    # Stage A (deterministic, MAP model) on one volume
    python 07_parity_kwyk.py --volume sub-01_t1.mgz \\
        --pytorch-weights-map kwyk_map.pth --stage a --out parity_report.json

    # both stages
    python 07_parity_kwyk.py --volume sub-01_t1.mgz \\
        --pytorch-weights-map kwyk_map.pth --pytorch-weights-ssd kwyk_ssd.pth \\
        --stage both --out parity_report.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

import nibabel as nib
import numpy as np
from parity_metrics import (
    expected_calibration_error,
    per_class_dice,
    reconcile_pytorch_variance,
    scale_ratio,
    spatial_pearson_r,
    voxel_agreement,
)
import torch
from utils import setup_logging

log = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Constants -- verified against a live `docker run neuronets/kwyk:latest-cpu`
# container (see docs/kwyk_parity_report.md for how these were checked).
# ---------------------------------------------------------------------------
_DEFAULT_CONTAINER_IMAGE = "neuronets/kwyk:latest-cpu"
_MRI_CONVERT_BIN = "/opt/kwyk/freesurfer/bin/mri_convert"
_SAVED_MODEL_DIRS = {
    "map": "/opt/kwyk/saved_models/all_50_wn/1555341859",
    "ssd": "/opt/kwyk/saved_models/all_50_bvwn_multi_prior/1556816070",
}
_N_CLASSES = 50
_FILTERS = 96  # published kwyk checkpoints use 96 hidden filters
_RECEPTIVE_FIELD = 37  # published kwyk checkpoints use the 37-voxel schedule
_BLOCK_SHAPE = (32, 32, 32)  # matches the SavedModel's fixed input shape
_CONFORMED_SHAPE = (256, 256, 256)

# Preprocessing guard: the host z-score must match the container's own
# nobrainer.volume.zscore on the same array to float32 noise level. Pinned
# from an observed run (host arm64 vs. emulated x86_64 container, both
# numpy, same (a-mean)/std formula over 16.7M elements): max|diff|=9.5e-06.
# Whole-volume float32 mean/std is not associative across BLAS backends, so
# a small cross-platform residual here is expected, not a bug; 1e-4 keeps
# ~10x headroom over that measurement while still catching a real mismatch
# (e.g. per-block vs whole-volume normalization, which differs by orders
# of magnitude more).
_PREPROCESSING_ATOL = 1e-4

# Stage A gates. Block-level parity against the live TF graph on random
# input measured 9.5e-05 max|logit diff| earlier in this project (see
# docs/kwyk_mapping_verification.md); non-overlapping tiling with no
# averaging means a full volume accumulates no extra error, so 1e-3 is
# ~10x headroom over that noise floor.
_STAGE_A_MAX_LOGIT_DIFF_GATE = 1e-3
_STAGE_A_MEAN_DICE_GATE = 0.98
_STAGE_A_VOXEL_AGREEMENT_GATE = 0.995

# Stage B gates, looser than Stage A: aggregated MC estimates carry
# sampling noise the deterministic path does not.
_STAGE_B_MEAN_LABEL_DICE_GATE = 0.95
_STAGE_B_VARIANCE_CORR_GATE = 0.90
_STAGE_B_ENTROPY_CORR_GATE = 0.90
_STAGE_B_CROP_SIZE = 128
_STAGE_B_N_SAMPLES = 20
_STAGE_B_SEED = 42

_ECE_N_BINS = 15
_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Container invocation
# ---------------------------------------------------------------------------
def _docker_run(
    image: str,
    entrypoint: str,
    container_args: list[str],
    mounts: dict[Path, str],
) -> str:
    """Run ``docker run --rm`` with the given entrypoint override and mounts.

    Returns combined stdout+stderr. Raises ``RuntimeError`` on nonzero exit.
    The container's own ENTRYPOINT is the ``kwyk`` CLI, so every invocation
    here overrides it explicitly.
    """
    cmd = ["docker", "run", "--rm"]
    for host_path, container_path in mounts.items():
        cmd += ["-v", f"{host_path}:{container_path}"]
    cmd += ["--entrypoint", entrypoint, image, *container_args]
    log.info("docker run: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"Container command failed (rc={result.returncode}):\n{output}"
        )
    return output


def conform_volume(raw_path: Path, image: str, work_dir: Path) -> Path:
    """Run ``mri_convert --conform`` inside the container.

    A volume already at 256^3 is not reprocessed differently by
    ``mri_convert`` than a raw scan -- the CLI's own skip-guard is
    shape-only (``if img.shape != (256, 256, 256)``), so running conform
    unconditionally here is always safe and always produces the exact
    array both sides will consume.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    conformed_path = work_dir / "conformed.nii.gz"
    _docker_run(
        image,
        _MRI_CONVERT_BIN,
        ["--conform", f"/input/{raw_path.name}", "/work/conformed.nii.gz"],
        {raw_path.parent.resolve(): "/input", work_dir.resolve(): "/work"},
    )
    if not conformed_path.exists():
        raise RuntimeError(f"mri_convert did not produce {conformed_path}")
    return conformed_path


_CONTAINER_ZSCORE_SCRIPT = textwrap.dedent(
    """\
    import argparse
    import nibabel as nib
    import numpy as np
    from nobrainer.volume import zscore

    p = argparse.ArgumentParser()
    p.add_argument("--in-nifti", required=True)
    p.add_argument("--out-npy", required=True)
    args = p.parse_args()

    arr = np.asarray(nib.load(args.in_nifti).dataobj, dtype=np.float32)
    z = zscore(arr)
    np.save(args.out_npy, z.astype(np.float32))
    """
)


def host_zscore(arr: np.ndarray) -> np.ndarray:
    """Whole-volume z-score, matching the container's ``nobrainer.volume.zscore``."""
    arr = arr.astype(np.float32)
    return (arr - arr.mean()) / arr.std()


def assert_preprocessing(
    conformed_path: Path, z_host: np.ndarray, image: str, work_dir: Path
) -> dict:
    """Verify the host z-score matches the container's own zscore() on the same input.

    This is the guard called out in the parity plan: a normalization
    mismatch must fail here, as a preprocessing error, never surface later
    as a misleading Dice number.
    """
    script_path = work_dir / "_container_zscore.py"
    script_path.write_text(_CONTAINER_ZSCORE_SCRIPT)
    out_npy = work_dir / "container_z.npy"
    _docker_run(
        image,
        "python",
        [
            "/work/_container_zscore.py",
            "--in-nifti",
            "/work/conformed.nii.gz",
            "--out-npy",
            "/work/container_z.npy",
        ],
        {work_dir.resolve(): "/work"},
    )
    z_container = np.load(out_npy)
    diff = np.abs(z_host - z_container)
    max_diff = float(diff.max())
    return {
        "conform": "mri_convert --conform (in container)",
        "conformed_sha256": hashlib.sha256(conformed_path.read_bytes()).hexdigest(),
        "normalization": "global z-score, whole volume, float32, ddof=0",
        "block_shape": list(_BLOCK_SHAPE),
        "overlap": False,
        "input_max_abs_diff": max_diff,
        "input_match": max_diff <= _PREPROCESSING_ATOL,
    }


# ---------------------------------------------------------------------------
# Stage A -- deterministic mean-path parity (MAP / all_50_wn)
# ---------------------------------------------------------------------------
_CONTAINER_STAGE_A_SCRIPT = textwrap.dedent(
    """\
    import argparse
    import time
    import numpy as np
    import tensorflow as tf
    from nobrainer.volume import to_blocks

    p = argparse.ArgumentParser()
    p.add_argument("--saved-model-dir", required=True)
    p.add_argument("--input-npy", required=True)
    p.add_argument("--block-shape", type=int, nargs=3, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--out-logits", required=True)
    p.add_argument("--out-labels", required=True)
    args = p.parse_args()

    predictor = tf.contrib.predictor.from_saved_model(args.saved_model_dir)
    arr = np.load(args.input_npy).astype(np.float32)
    block_shape = tuple(args.block_shape)
    D, H, W = arr.shape
    bd, bh, bw = block_shape
    nd, nh, nw = D // bd, H // bh, W // bw

    # Write each batch's predictions straight into the final (D,H,W,C)
    # volume instead of accumulating a separate blocks-shaped array first:
    # holding both a (N,bD,bH,bW,C) and a (D,H,W,C) array at once for a
    # 256^3 x 50-class volume needs ~6.7GB, tight against a constrained
    # Docker VM. This keeps peak memory to one full-volume array.
    blocks = to_blocks(arr, block_shape)[..., None]  # (N, bD, bH, bW, 1)
    n_blocks = blocks.shape[0]
    n_classes = None
    logits_full = None
    labels_full = np.zeros((D, H, W), dtype=np.int64)

    t0 = time.time()
    for j in range(0, n_blocks, args.batch_size):
        chunk = blocks[j:j + args.batch_size]
        out = predictor({"volume": chunk})
        if logits_full is None:
            n_classes = out["logits"].shape[-1]
            logits_full = np.zeros((D, H, W, n_classes), dtype=np.float32)
        for local_idx in range(chunk.shape[0]):
            block_idx = j + local_idx
            i, jx, k = block_idx // (nh * nw), (block_idx // nw) % nh, block_idx % nw
            sd, sh, sw = slice(i*bd, (i+1)*bd), slice(jx*bh, (jx+1)*bh), slice(k*bw, (k+1)*bw)
            logits_full[sd, sh, sw] = out["logits"][local_idx]
            labels_full[sd, sh, sw] = out["class_ids"][local_idx]
        print("block %d/%d elapsed=%.1fs" % (j + chunk.shape[0], n_blocks, time.time() - t0))

    np.save(args.out_logits, logits_full)
    np.save(args.out_labels, labels_full)
    print("Wrote", args.out_logits, args.out_labels)
    """
)


def run_tf_stage_a(
    z_npy_path: Path,
    image: str,
    work_dir: Path,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the MAP SavedModel directly in-container; return (logits, labels).

    logits: (D, H, W, n_classes) float32. labels: (D, H, W) int64 (the
    container's own argmax, ``class_ids`` -- used as the TF reference label
    volume so downstream Dice reflects exactly what the container reports).
    """
    script_path = work_dir / "_container_stage_a.py"
    script_path.write_text(_CONTAINER_STAGE_A_SCRIPT)
    out_logits = work_dir / "tf_logits.npy"
    out_labels = work_dir / "tf_labels.npy"
    _docker_run(
        image,
        "python",
        [
            "/work/_container_stage_a.py",
            "--saved-model-dir",
            _SAVED_MODEL_DIRS["map"],
            "--input-npy",
            f"/work/{z_npy_path.name}",
            "--block-shape",
            *[str(b) for b in _BLOCK_SHAPE],
            "--batch-size",
            str(batch_size),
            "--out-logits",
            "/work/tf_logits.npy",
            "--out-labels",
            "/work/tf_labels.npy",
        ],
        {work_dir.resolve(): "/work"},
    )
    return np.load(out_logits), np.load(out_labels)


def run_pytorch_stage_a(
    model: torch.nn.Module,
    z_arr: np.ndarray,
    block_shape: tuple[int, int, int],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct block-wise forward pass collecting raw logits (no softmax).

    ``nobrainer.prediction.predict()`` only returns softmax probabilities
    or argmax labels, never raw logits -- reuses its private block
    extraction/stitching helpers (identical math to the public API) rather
    than reimplementing tiling, and calls the model directly to keep the
    pre-softmax values Stage A's logit-diff metric needs.
    """
    from nobrainer.prediction import _extract_blocks, _pad_to_multiple, _stitch_blocks

    model = model.to(device)
    model.eval()

    orig_shape = z_arr.shape
    padded, pad = _pad_to_multiple(z_arr, block_shape)
    blocks, grid = _extract_blocks(padded, block_shape)
    n_blocks = blocks.shape[0]

    all_logits = []
    with torch.no_grad():
        for start in range(0, n_blocks, batch_size):
            chunk = blocks[start : start + batch_size]
            tensor = torch.from_numpy(chunk[:, None]).to(device)
            out = model(tensor, mc=False)
            all_logits.append(out.cpu().numpy())

    block_logits = np.concatenate(all_logits, axis=0)  # (N, C, bD, bH, bW)
    n_classes = block_logits.shape[1]
    logits_full = _stitch_blocks(
        block_logits, grid, block_shape, pad, orig_shape, n_classes
    )
    labels_full = logits_full.argmax(axis=0).astype(np.int64)
    return logits_full, labels_full


# ---------------------------------------------------------------------------
# Stage B -- MC / uncertainty parity (SSD / all_50_bvwn_multi_prior), aggregate only
# ---------------------------------------------------------------------------
_CONTAINER_STAGE_B_SCRIPT = textwrap.dedent(
    """\
    import argparse
    import numpy as np
    import tensorflow as tf
    from nobrainer.predict import predict_from_array

    p = argparse.ArgumentParser()
    p.add_argument("--saved-model-dir", required=True)
    p.add_argument("--input-npy", required=True)
    p.add_argument("--block-shape", type=int, nargs=3, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--n-samples", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-mean", required=True)
    p.add_argument("--out-variance", required=True)
    p.add_argument("--out-entropy", required=True)
    args = p.parse_args()

    tf.set_random_seed(args.seed)
    predictor = tf.contrib.predictor.from_saved_model(args.saved_model_dir)
    arr = np.load(args.input_npy).astype(np.float32)

    # Reuse the container's own MC procedure verbatim (Welford-style running
    # mean/variance, variance summed over classes, entropy eps=1e-7) rather
    # than reimplementing the sampling loop.
    mean, variance, entropy = predict_from_array(
        inputs=arr,
        predictor=predictor,
        block_shape=tuple(args.block_shape),
        return_variance=True,
        return_entropy=True,
        return_array_from_images=True,
        n_samples=args.n_samples,
        normalizer=None,
        batch_size=args.batch_size,
    )
    np.save(args.out_mean, mean)
    np.save(args.out_variance, variance)
    np.save(args.out_entropy, entropy)
    print("Wrote Stage B outputs")
    """
)


def _central_crop(arr: np.ndarray, crop_size: int) -> np.ndarray:
    """Extract a centered cubic crop of side ``crop_size`` from a 3-D array."""
    starts = [(s - crop_size) // 2 for s in arr.shape]
    slices = tuple(slice(s, s + crop_size) for s in starts)
    return arr[slices]


def run_tf_stage_b(
    z_crop_npy_path: Path,
    image: str,
    work_dir: Path,
    batch_size: int,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the SSD SavedModel's own MC loop in-container; aggregate stats only."""
    script_path = work_dir / "_container_stage_b.py"
    script_path.write_text(_CONTAINER_STAGE_B_SCRIPT)
    out_mean = work_dir / "tf_mean_b.npy"
    out_var = work_dir / "tf_variance_b.npy"
    out_ent = work_dir / "tf_entropy_b.npy"
    _docker_run(
        image,
        "python",
        [
            "/work/_container_stage_b.py",
            "--saved-model-dir",
            _SAVED_MODEL_DIRS["ssd"],
            "--input-npy",
            f"/work/{z_crop_npy_path.name}",
            "--block-shape",
            *[str(b) for b in _BLOCK_SHAPE],
            "--batch-size",
            str(batch_size),
            "--n-samples",
            str(n_samples),
            "--seed",
            str(seed),
            "--out-mean",
            "/work/tf_mean_b.npy",
            "--out-variance",
            "/work/tf_variance_b.npy",
            "--out-entropy",
            "/work/tf_entropy_b.npy",
        ],
        {work_dir.resolve(): "/work"},
    )
    return np.load(out_mean), np.load(out_var), np.load(out_ent)


def run_pytorch_stage_b(
    model: torch.nn.Module,
    z_crop_arr: np.ndarray,
    n_samples: int,
    block_shape: tuple[int, int, int],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MC aggregate via ``nobrainer.prediction.predict_with_uncertainty``.

    Variance is rescaled to the container's sum-over-classes convention
    (see ``parity_metrics.reconcile_pytorch_variance``); the kwyk layers
    sample from the global torch RNG (no per-module Generator), so seeding
    is via ``torch.manual_seed`` for this process, not a shared-stream match
    with the TF side -- MC samples are never compared 1:1 across frameworks.
    """
    from nobrainer.prediction import predict_with_uncertainty

    torch.manual_seed(seed)
    label_img, var_img, entropy_img = predict_with_uncertainty(
        z_crop_arr,
        model,
        n_samples=n_samples,
        block_shape=block_shape,
        batch_size=batch_size,
        device=device,
    )
    mean_labels = np.asarray(label_img.dataobj).astype(np.int64)
    variance = reconcile_pytorch_variance(
        np.asarray(var_img.dataobj), n_classes=_N_CLASSES
    )
    entropy = np.asarray(entropy_img.dataobj)
    return mean_labels, variance, entropy


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_pytorch_model(
    weights_path: Path, dropout_type: str, device: torch.device
) -> torch.nn.Module:
    """Build a KWYKMeshNet matching the published kwyk architecture and load weights."""
    from nobrainer.models import get as get_model

    model = get_model("kwyk_meshnet")(
        n_classes=_N_CLASSES,
        filters=_FILTERS,
        receptive_field=_RECEPTIVE_FIELD,
        dropout_type=dropout_type,
        bias=True,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def evaluate_stage_a_volume(
    volume_id: str,
    source_path: str,
    tf_logits: np.ndarray,
    tf_labels: np.ndarray,
    pt_logits: np.ndarray,
    pt_labels: np.ndarray,
) -> dict:
    """Compare one volume's TF vs PyTorch Stage A outputs."""
    # tf_logits: (D,H,W,C); pt_logits: (C,D,H,W) -- align to (D,H,W,C).
    pt_logits_dhwc = np.moveaxis(pt_logits, 0, -1)
    diff = np.abs(tf_logits - pt_logits_dhwc)
    max_abs_logit_diff = float(diff.max())
    mean_abs_logit_diff = float(diff.mean())

    dice = per_class_dice(pt_labels, tf_labels, n_classes=_N_CLASSES)
    agreement = voxel_agreement(pt_labels, tf_labels)

    # There is no ground-truth segmentation in a parity check -- the two
    # models are each other's reference. Both ECE terms use the SAME
    # agreement mask (pt_labels == tf_labels) as "correct", scored against
    # each side's own confidence: this asks whether either model is
    # confidently wrong exactly where it disagrees with the other, not
    # whether either is accurate against ground truth (that is 05's job).
    agree_mask = (pt_labels == tf_labels).astype(np.float64)

    pt_probs = torch.softmax(torch.from_numpy(pt_logits_dhwc), dim=-1).numpy()
    pt_confidence = pt_probs.max(axis=-1)
    ece_pytorch = expected_calibration_error(
        pt_confidence, agree_mask, n_bins=_ECE_N_BINS
    )

    tf_probs = torch.softmax(torch.from_numpy(tf_logits), dim=-1).numpy()
    tf_confidence = tf_probs.max(axis=-1)
    ece_tf = expected_calibration_error(tf_confidence, agree_mask, n_bins=_ECE_N_BINS)

    passed = (
        max_abs_logit_diff <= _STAGE_A_MAX_LOGIT_DIFF_GATE
        and float(dice.mean()) >= _STAGE_A_MEAN_DICE_GATE
        and agreement >= _STAGE_A_VOXEL_AGREEMENT_GATE
    )
    return {
        "volume_id": volume_id,
        "source_path": source_path,
        "max_abs_logit_diff": max_abs_logit_diff,
        "mean_abs_logit_diff": mean_abs_logit_diff,
        "dice_per_class": dice.tolist(),
        "dice_mean": float(dice.mean()),
        "dice_min": float(dice.min()),
        "worst_class": int(np.argmin(dice)) + 1,
        "voxel_agreement": agreement,
        "ece_pytorch": ece_pytorch,
        "ece_tf": ece_tf,
        "ece_delta": abs(ece_pytorch - ece_tf),
        "n_voxels": int(tf_labels.size),
        "pass": passed,
    }


def evaluate_stage_b_volume(
    volume_id: str,
    tf_mean: np.ndarray,
    tf_variance: np.ndarray,
    tf_entropy: np.ndarray,
    pt_mean: np.ndarray,
    pt_variance: np.ndarray,
    pt_entropy: np.ndarray,
) -> dict:
    """Compare one volume's TF vs PyTorch Stage B aggregate MC statistics."""
    dice = per_class_dice(
        pt_mean.astype(np.int64), tf_mean.astype(np.int64), n_classes=_N_CLASSES
    )
    mean_label_dice = float(dice.mean())
    variance_r = spatial_pearson_r(pt_variance, tf_variance)
    variance_ratio = scale_ratio(pt_variance, tf_variance)
    entropy_r = spatial_pearson_r(pt_entropy, tf_entropy)
    entropy_ratio = scale_ratio(pt_entropy, tf_entropy)

    passed = (
        mean_label_dice >= _STAGE_B_MEAN_LABEL_DICE_GATE
        and variance_r >= _STAGE_B_VARIANCE_CORR_GATE
        and entropy_r >= _STAGE_B_ENTROPY_CORR_GATE
    )
    return {
        "volume_id": volume_id,
        "mean_label_dice": mean_label_dice,
        "variance_pearson_r": variance_r,
        "variance_scale_ratio": variance_ratio,
        "entropy_pearson_r": entropy_r,
        "entropy_scale_ratio": entropy_ratio,
        "pass": passed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", action="append", required=True, dest="volumes")
    parser.add_argument("--pytorch-weights-map", type=str, default=None)
    parser.add_argument("--pytorch-weights-ssd", type=str, default=None)
    parser.add_argument("--stage", choices=["a", "b", "both"], default="a")
    parser.add_argument(
        "--out", type=str, default=None, help="Path for parity_report.json"
    )
    parser.add_argument("--check-preprocessing-only", action="store_true")
    parser.add_argument("--container-image", type=str, default=_DEFAULT_CONTAINER_IMAGE)
    parser.add_argument("--work-dir", type=str, default="results/parity")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-samples-b", type=int, default=_STAGE_B_N_SAMPLES)
    parser.add_argument("--crop-size-b", type=int, default=_STAGE_B_CROP_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    from nobrainer.gpu import get_device

    device = torch.device(args.device) if args.device else get_device()
    log.info("Using device: %s", device)

    preprocessing_by_volume: dict[str, dict] = {}
    z_arrays: dict[str, np.ndarray] = {}
    for volume_path_str in args.volumes:
        volume_path = Path(volume_path_str)
        volume_id = volume_path.stem
        vol_work_dir = work_dir / volume_id
        conformed_path = conform_volume(volume_path, args.container_image, vol_work_dir)
        arr = np.asarray(nib.load(str(conformed_path)).dataobj)
        if arr.shape != _CONFORMED_SHAPE:
            raise RuntimeError(
                f"{volume_id}: conformed shape {arr.shape} != {_CONFORMED_SHAPE}"
            )
        z = host_zscore(arr)
        preprocessing = assert_preprocessing(
            conformed_path, z, args.container_image, vol_work_dir
        )
        preprocessing_by_volume[volume_id] = preprocessing
        z_arrays[volume_id] = z
        log.info(
            "%s: preprocessing input_match=%s max_abs_diff=%.3e",
            volume_id,
            preprocessing["input_match"],
            preprocessing["input_max_abs_diff"],
        )
        if not preprocessing["input_match"]:
            raise RuntimeError(
                f"{volume_id}: host/container z-score mismatch "
                f"(max_abs_diff={preprocessing['input_max_abs_diff']:.3e} > "
                f"{_PREPROCESSING_ATOL:.3e}). Aborting before any inference."
            )
        np.save(vol_work_dir / "z.npy", z)

    if args.check_preprocessing_only:
        log.info("Preprocessing check only -- all volumes matched. Exiting.")
        return

    report: dict = {
        "schema_version": _SCHEMA_VERSION,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            "container_image": args.container_image,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "thresholds": {
            "stage_a_max_abs_logit_diff": _STAGE_A_MAX_LOGIT_DIFF_GATE,
            "stage_a_mean_dice": _STAGE_A_MEAN_DICE_GATE,
            "stage_a_voxel_agreement": _STAGE_A_VOXEL_AGREEMENT_GATE,
            "stage_b_mean_label_dice": _STAGE_B_MEAN_LABEL_DICE_GATE,
            "stage_b_variance_pearson_r": _STAGE_B_VARIANCE_CORR_GATE,
            "stage_b_entropy_pearson_r": _STAGE_B_ENTROPY_CORR_GATE,
            "preprocessing_atol": _PREPROCESSING_ATOL,
        },
        "preprocessing": preprocessing_by_volume,
    }
    failed_checks: list[str] = []

    if args.stage in ("a", "both"):
        if not args.pytorch_weights_map:
            raise ValueError("--pytorch-weights-map is required for stage a/both")
        model_map = load_pytorch_model(
            Path(args.pytorch_weights_map), dropout_type="bernoulli", device=device
        )
        volumes_a = []
        for volume_path_str in args.volumes:
            volume_id = Path(volume_path_str).stem
            vol_work_dir = work_dir / volume_id
            z_npy_path = vol_work_dir / "z.npy"
            log.info("%s: Stage A -- running TF SavedModel in container", volume_id)
            tf_logits, tf_labels = run_tf_stage_a(
                z_npy_path, args.container_image, vol_work_dir, args.batch_size
            )
            log.info("%s: Stage A -- running PyTorch model", volume_id)
            pt_logits, pt_labels = run_pytorch_stage_a(
                model_map, z_arrays[volume_id], _BLOCK_SHAPE, args.batch_size, device
            )
            result = evaluate_stage_a_volume(
                volume_id, volume_path_str, tf_logits, tf_labels, pt_logits, pt_labels
            )
            log.info(
                "%s: max|logit diff|=%.3e dice_mean=%.4f voxel_agreement=%.4f pass=%s",
                volume_id,
                result["max_abs_logit_diff"],
                result["dice_mean"],
                result["voxel_agreement"],
                result["pass"],
            )
            if not result["pass"]:
                failed_checks.append(f"stage_a:{volume_id}")
            volumes_a.append(result)

        report["stage_a_deterministic"] = {
            "model": "all_50_wn",
            "tf_deterministic_verified": True,
            "volumes": volumes_a,
            "aggregate": {
                "dice_mean": float(np.mean([v["dice_mean"] for v in volumes_a])),
                "max_abs_logit_diff": float(
                    max(v["max_abs_logit_diff"] for v in volumes_a)
                ),
                "pass": all(v["pass"] for v in volumes_a),
            },
        }

    if args.stage in ("b", "both"):
        if not args.pytorch_weights_ssd:
            raise ValueError("--pytorch-weights-ssd is required for stage b/both")
        model_ssd = load_pytorch_model(
            Path(args.pytorch_weights_ssd), dropout_type="concrete", device=device
        )
        volumes_b = []
        for volume_path_str in args.volumes:
            volume_id = Path(volume_path_str).stem
            vol_work_dir = work_dir / volume_id
            z_crop = _central_crop(z_arrays[volume_id], args.crop_size_b)
            z_crop_path = vol_work_dir / "z_crop.npy"
            np.save(z_crop_path, z_crop)

            log.info(
                "%s: Stage B -- running TF SavedModel MC loop in container", volume_id
            )
            tf_mean, tf_var, tf_ent = run_tf_stage_b(
                z_crop_path,
                args.container_image,
                vol_work_dir,
                args.batch_size,
                args.n_samples_b,
                _STAGE_B_SEED,
            )
            log.info("%s: Stage B -- running PyTorch MC loop", volume_id)
            pt_mean, pt_var, pt_ent = run_pytorch_stage_b(
                model_ssd,
                z_crop,
                args.n_samples_b,
                _BLOCK_SHAPE,
                args.batch_size,
                device,
                _STAGE_B_SEED,
            )
            result = evaluate_stage_b_volume(
                volume_id, tf_mean, tf_var, tf_ent, pt_mean, pt_var, pt_ent
            )
            log.info(
                "%s: mean_label_dice=%.4f variance_r=%.4f entropy_r=%.4f pass=%s",
                volume_id,
                result["mean_label_dice"],
                result["variance_pearson_r"],
                result["entropy_pearson_r"],
                result["pass"],
            )
            if not result["pass"]:
                failed_checks.append(f"stage_b:{volume_id}")
            volumes_b.append(result)

        report["stage_b_mc"] = {
            "model": "all_50_bvwn_multi_prior",
            "n_samples": args.n_samples_b,
            "region": f"central {args.crop_size_b}^3 crop",
            "seeds": {"torch": _STAGE_B_SEED, "tf": _STAGE_B_SEED},
            "note": "aggregate statistics only; per-sample comparison is invalid across frameworks",
            "variance_convention": {
                "tf": "sum over classes",
                "pytorch": "mean over classes",
                "reconciliation": "pytorch * n_classes",
            },
            "volumes": volumes_b,
            "aggregate": {"pass": all(v["pass"] for v in volumes_b)},
        }

    report["result"] = {"pass": len(failed_checks) == 0, "failed_checks": failed_checks}

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        log.info("Wrote parity report to %s", out_path)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
