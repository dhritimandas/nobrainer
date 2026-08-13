"""Metric helpers for kwyk PyTorch-vs-container parity checks.

Used by ``07_parity_kwyk.py``. Kept separate and dependency-light (numpy
only) so the metrics are unit-testable without a container or GPU.
"""

from __future__ import annotations

import numpy as np


def per_class_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Compute Dice coefficient for each class c = 1..n_classes-1.

    Duplicated from ``04_evaluate.py`` (not reimported from it): that module
    does a top-level ``import matplotlib.pyplot``, an undeclared dependency
    not installed in this environment, and pulling in a plotting import to
    reuse a 12-line numeric function is not worth adding a new dependency
    for. Matches Eq. 19 in McClure et al. (2019); keep in sync with
    ``04_evaluate.per_class_dice`` if that formula ever changes.

    Class 0 (background / unknown) is excluded, matching the paper.

    Parameters
    ----------
    pred : np.ndarray
        Integer label predictions.
    gt : np.ndarray
        Integer ground truth labels.
    n_classes : int
        Total number of classes (including background).

    Returns
    -------
    np.ndarray
        Shape ``(n_classes - 1,)`` — Dice for classes 1..n_classes-1.
    """
    dice_scores = np.zeros(n_classes - 1)
    for c in range(1, n_classes):
        pred_c = (pred == c).astype(np.float64)
        gt_c = (gt == c).astype(np.float64)
        intersection = (pred_c * gt_c).sum()
        total = pred_c.sum() + gt_c.sum()
        if total > 0:
            dice_scores[c - 1] = 2.0 * intersection / total
        else:
            dice_scores[c - 1] = 1.0
    return dice_scores


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Top-1 expected calibration error (ECE).

    Parameters
    ----------
    confidence : np.ndarray
        Per-voxel max predicted probability, flattened, in ``[0, 1]``.
    correct : np.ndarray
        Per-voxel boolean/0-1 array: whether the argmax prediction matches
        the reference label. Same shape as ``confidence``.
    n_bins : int
        Number of equal-width confidence bins.

    Returns
    -------
    float
        ECE in ``[0, 1]``: the confidence-weighted average gap between
        predicted confidence and observed accuracy across bins.
    """
    confidence = confidence.ravel().astype(np.float64)
    correct = correct.ravel().astype(np.float64)
    if confidence.size == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, bin_edges[1:-1]), 0, n_bins - 1)

    ece = 0.0
    n_total = confidence.size
    for b in range(n_bins):
        mask = bin_idx == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        bin_confidence = confidence[mask].mean()
        bin_accuracy = correct[mask].mean()
        ece += (n_bin / n_total) * abs(bin_confidence - bin_accuracy)
    return float(ece)


def voxel_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Fraction of voxels where two integer label volumes agree."""
    if labels_a.shape != labels_b.shape:
        raise ValueError(f"Shape mismatch: {labels_a.shape} vs {labels_b.shape}")
    return float(np.mean(labels_a == labels_b))


def spatial_pearson_r(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """Pearson correlation between two spatial maps of the same shape.

    Returns 0.0 if either map is constant (undefined correlation).
    """
    a = map_a.ravel().astype(np.float64)
    b = map_b.ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def scale_ratio(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """Ratio of mean(map_a) to mean(map_b); 0.0 if the denominator is 0."""
    denom = float(map_b.mean())
    if denom == 0.0:
        return 0.0
    return float(map_a.mean()) / denom


# The container's variance convention sums over classes
# (nobrainer/predict.py: ``np.sum(M / n_samples, axis=-1)``); PyTorch's
# ``predict_with_uncertainty`` averages over classes instead
# (nobrainer/prediction.py: ``var_probs.mean(axis=1)``). Multiplying by
# n_classes converts the PyTorch convention to the TF one so the two are
# comparable on the same scale.
def reconcile_pytorch_variance(
    pytorch_mean_variance: np.ndarray, n_classes: int
) -> np.ndarray:
    """Rescale PyTorch's mean-over-classes variance to TF's sum-over-classes."""
    return pytorch_mean_variance * n_classes


__all__ = [
    "per_class_dice",
    "expected_calibration_error",
    "voxel_agreement",
    "spatial_pearson_r",
    "scale_ratio",
    "reconcile_pytorch_variance",
]
