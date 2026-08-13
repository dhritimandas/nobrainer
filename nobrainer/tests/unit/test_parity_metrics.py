"""Unit tests for scripts/kwyk_reproduction/parity_metrics.py.

Hermetic: no container, no network, no GPU. Loaded via importlib because
the module lives outside the ``nobrainer`` package (it is a script-support
module for the kwyk reproduction pipeline).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "kwyk_reproduction"
    / "parity_metrics.py"
)
_spec = importlib.util.spec_from_file_location("parity_metrics", _MODULE_PATH)
parity_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity_metrics)


class TestExpectedCalibrationError:
    def test_perfect_calibration_is_zero(self) -> None:
        # confidence == accuracy in every bin -> ECE == 0
        confidence = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        correct = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])  # 90% accurate
        ece = parity_metrics.expected_calibration_error(confidence, correct, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_fully_overconfident_wrong_gives_max_ece(self) -> None:
        confidence = np.full(100, 1.0)
        correct = np.zeros(100)
        ece = parity_metrics.expected_calibration_error(confidence, correct, n_bins=10)
        assert ece == pytest.approx(1.0, abs=1e-9)

    def test_empty_input_returns_zero(self) -> None:
        ece = parity_metrics.expected_calibration_error(
            np.array([]), np.array([]), n_bins=10
        )
        assert ece == 0.0


class TestVoxelAgreement:
    def test_identical_arrays_agree_fully(self) -> None:
        labels = np.random.randint(0, 50, size=(8, 8, 8))
        assert parity_metrics.voxel_agreement(labels, labels) == 1.0

    def test_half_disagreement(self) -> None:
        a = np.zeros(10, dtype=np.int64)
        b = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
        assert parity_metrics.voxel_agreement(a, b) == pytest.approx(0.5)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            parity_metrics.voxel_agreement(np.zeros((2, 2)), np.zeros((3, 3)))


class TestSpatialPearsonR:
    def test_identical_maps_give_r_one(self) -> None:
        rng = np.random.default_rng(0)
        m = rng.normal(size=(16, 16, 16))
        assert parity_metrics.spatial_pearson_r(m, m) == pytest.approx(1.0, abs=1e-9)

    def test_anticorrelated_maps_give_r_minus_one(self) -> None:
        rng = np.random.default_rng(0)
        m = rng.normal(size=(16, 16, 16))
        assert parity_metrics.spatial_pearson_r(m, -m) == pytest.approx(-1.0, abs=1e-9)

    def test_constant_map_returns_zero(self) -> None:
        m1 = np.full((4, 4, 4), 5.0)
        m2 = np.random.default_rng(0).normal(size=(4, 4, 4))
        assert parity_metrics.spatial_pearson_r(m1, m2) == 0.0


class TestScaleRatio:
    def test_double_scale(self) -> None:
        a = np.full(10, 2.0)
        b = np.full(10, 1.0)
        assert parity_metrics.scale_ratio(a, b) == pytest.approx(2.0)

    def test_zero_denominator_returns_zero(self) -> None:
        a = np.full(10, 2.0)
        b = np.zeros(10)
        assert parity_metrics.scale_ratio(a, b) == 0.0


class TestReconcilePytorchVariance:
    def test_multiplies_by_n_classes(self) -> None:
        var = np.full((4, 4, 4), 0.5)
        reconciled = parity_metrics.reconcile_pytorch_variance(var, n_classes=50)
        np.testing.assert_allclose(reconciled, np.full((4, 4, 4), 25.0))


class TestPerClassDice:
    def test_reused_from_04_evaluate(self) -> None:
        # Perfect agreement across 3 foreground classes -> Dice == 1 for each.
        pred = np.array([1, 2, 3, 0])
        gt = np.array([1, 2, 3, 0])
        dice = parity_metrics.per_class_dice(pred, gt, n_classes=4)
        np.testing.assert_allclose(dice, [1.0, 1.0, 1.0])

    def test_absent_class_scores_perfect_agreement(self) -> None:
        pred = np.array([0, 0, 0])
        gt = np.array([0, 0, 0])
        dice = parity_metrics.per_class_dice(pred, gt, n_classes=2)
        np.testing.assert_allclose(dice, [1.0])
