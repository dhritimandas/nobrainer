"""Golden-output regression tests for ``nobrainer.prediction.predict()``.

Pins the inference output of a tiny, hermetic brain-extraction UNet against a
committed fixture (``golden/golden_brain_extraction.npz``) so a refactor to
``predict()``'s padding, block splitting, reassembly, or argmax/threshold logic
cannot silently change what users get. No network access, no GPU, no downloads
-- the input volume and label are regenerated analytically from the fixture's
stored ``meta``, and the model is the committed weights.

Every tolerance below is justified where it is used; see ``golden/README.md``
for the full rationale and for how to deliberately regenerate the fixture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from monai.utils import set_determinism
import nibabel as nib
import numpy as np
import pytest
import torch

from nobrainer.models import get as get_model
from nobrainer.prediction import predict

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "golden" / "golden_brain_extraction.npz"
)
SEED = 42


# ---------------------------------------------------------------------------
# Helpers duplicated (not imported) from golden/generate_golden.py.
#
# nobrainer/sr-tests is not a dotted-importable package (hyphen in the name),
# so this test cannot `from golden.generate_golden import ...`. Both copies
# are pure functions of the fixture's stored `meta` values (volume_shape,
# sphere_radius, n_probe_voxels) with no hidden constants, so they cannot
# silently diverge without a visible diff to both files in the same PR.
# ---------------------------------------------------------------------------


def _make_sphere_volume(
    shape: tuple[int, int, int], radius: float
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]].astype(np.float32)
    center = np.array(shape, dtype=np.float32) / 2
    dist = np.sqrt(sum((coords[i] - center[i]) ** 2 for i in range(3)))
    label = (dist < radius).astype(np.float32)
    vol = np.clip(1.0 - dist / dist.max(), 0.0, 1.0).astype(np.float32) * 0.3
    vol = vol + label * 0.7
    return vol, label


def _probe_indices(volume_shape: tuple[int, int, int], n_probe: int) -> np.ndarray:
    total = int(np.prod(volume_shape))
    stride = total // n_probe
    return (np.arange(n_probe) * stride).astype(np.int64)


def _sha256_of_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _compare_label_masks(
    actual: np.ndarray, golden: np.ndarray, *, context: str = ""
) -> None:
    """sha256 fast path, falling back to a disagreement-budget + Dice check.

    Factored out so the tolerance logic itself can be unit-tested directly
    against a synthetically perturbed mask, independent of predict().
    """
    actual_flat = actual.ravel().astype(np.uint8)
    golden_flat = golden.ravel().astype(np.uint8)

    actual_sha = _sha256_of_array(np.packbits(actual_flat))
    golden_sha = _sha256_of_array(np.packbits(golden_flat))

    # Reassembly on the default (non-strided) prediction path is pure block
    # assignment with no float accumulation (nobrainer/prediction.py:60-79),
    # so on a fixed platform + device="cpu" the label mask is genuinely
    # bit-stable. This exact match is the primary regression signal: a
    # mismatch here means something in the pipeline actually changed, not
    # float noise.
    if actual_sha == golden_sha:
        return

    total = golden_flat.size
    n_diff = int(np.sum(actual_flat != golden_flat))
    intersection = float(
        np.sum(actual_flat.astype(np.float32) * golden_flat.astype(np.float32))
    )
    dice = 2 * intersection / (actual_flat.sum() + golden_flat.sum() + 1e-8)

    # Fallback tolerance: brain masks are near-binary, so the only
    # *legitimate* cross-platform disagreement is a boundary voxel where the
    # two class logits are within float noise of each other and argmax tips
    # the other way -- those exist only on the mask surface. This fixture's
    # synthetic sphere (radius 12) has 6,865 positive voxels out of 107,520
    # (6.4%); 0.05% of the volume (53 voxels) is generous headroom for
    # last-ULP boundary noise relative to that, while a real structural
    # change -- a shifted block, an off-by-one pad, a flipped axis -- moves
    # voxels throughout the volume, not just its surface, and would still
    # fail this budget.
    max_allowed_diff = int(0.0005 * total)
    assert n_diff <= max_allowed_diff, (
        f"{context}sha256 mismatch AND {n_diff}/{total} label voxels differ "
        f"(budget {max_allowed_diff}); dice={dice:.6f}. "
        "See nobrainer/sr-tests/golden/README.md before regenerating."
    )

    # Second, independent measure alongside the raw voxel-count budget: a
    # count-only check can be satisfied by a change that moves few voxels
    # but moves them systematically (e.g. a consistent 1-voxel shift along
    # one face), which the count budget alone would let through. Dice is the
    # metric this domain actually reads, so it must independently agree.
    #
    # 0.995, not something closer to 1.0: this fixture's sphere is small
    # (6,865 of 107,520 voxels, 6.4%), so Dice's denominator is the *object*
    # size, not the volume -- measured directly, 53 flipped voxels (the
    # voxel-count budget's ceiling) already pushes Dice down to ~0.9962 in
    # the worst case (all false positives or all false negatives, none
    # cancelling). A floor near 0.9995 would make this check fire before the
    # voxel-count budget ever reached its own ceiling, silently making Dice
    # the real (and wrongly strict) bottleneck. 0.995 sits just under that
    # measured worst-case-at-budget value.
    assert dice >= 0.995, (
        f"{context}label disagreement within the voxel-count budget "
        f"({n_diff}/{total}) but dice={dice:.6f} < 0.995 -- the differing "
        "voxels are not randomly scattered noise."
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden() -> tuple[np.lib.npyio.NpzFile, dict[str, Any]]:
    data = np.load(FIXTURE_PATH, allow_pickle=False)
    meta = json.loads(bytes(data["meta"]).decode("utf-8"))
    return data, meta


@pytest.fixture(scope="module")
def golden_model(golden) -> torch.nn.Module:
    data, meta = golden
    arch = meta["arch_kwargs"]
    model = get_model("unet")(
        n_classes=arch["n_classes"],
        in_channels=arch["in_channels"],
        channels=tuple(arch["channels"]),
        strides=tuple(arch["strides"]),
    )
    state_dict = {
        k[len("weights__") :]: torch.from_numpy(data[k])
        for k in data.files
        if k.startswith("weights__")
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@pytest.fixture(scope="module")
def golden_volume(golden) -> tuple[np.ndarray, np.ndarray]:
    _, meta = golden
    return _make_sphere_volume(tuple(meta["volume_shape"]), meta["sphere_radius"])


@pytest.fixture(scope="module")
def golden_prediction(golden, golden_model, golden_volume) -> nib.Nifti1Image:
    _, meta = golden
    vol, _ = golden_volume
    # set_determinism before every run that invokes the model -- see
    # golden/generate_golden.py for why this is defensive rather than
    # strictly load-bearing for this particular architecture.
    set_determinism(seed=SEED)
    golden_model.eval()
    return predict(
        inputs=vol,
        model=golden_model,
        block_shape=tuple(meta["block_shape"]),
        batch_size=meta["batch_size"],
        device="cpu",
        return_labels=True,
    )


# ---------------------------------------------------------------------------
# The tolerance logic itself, tested directly against a perturbed mask
# ---------------------------------------------------------------------------


class TestCompareLabelMasksTolerance:
    def test_identical_masks_pass_via_sha256(self) -> None:
        mask = (np.arange(1000) % 3 == 0).astype(np.float32)
        _compare_label_masks(mask, mask.copy())

    def test_within_budget_perturbation_passes(self, golden) -> None:
        data, meta = golden
        n_total = int(np.prod(meta["volume_shape"]))
        golden_labels = np.unpackbits(data["labels_packed"])[:n_total].astype(
            np.float32
        )

        perturbed = golden_labels.copy()
        n_flip = int(0.0003 * n_total)  # under the 0.05% budget
        flip_idx = np.arange(n_flip) * (n_total // n_flip)
        perturbed[flip_idx] = 1.0 - perturbed[flip_idx]

        _compare_label_masks(perturbed, golden_labels, context="[synthetic] ")

    def test_over_budget_perturbation_fails(self, golden) -> None:
        data, meta = golden
        n_total = int(np.prod(meta["volume_shape"]))
        golden_labels = np.unpackbits(data["labels_packed"])[:n_total].astype(
            np.float32
        )

        perturbed = golden_labels.copy()
        n_flip = int(0.01 * n_total)  # 20x over the 0.05% budget
        flip_idx = np.arange(n_flip) * (n_total // n_flip)
        perturbed[flip_idx] = 1.0 - perturbed[flip_idx]

        with pytest.raises(AssertionError):
            _compare_label_masks(perturbed, golden_labels, context="[synthetic] ")


# ---------------------------------------------------------------------------
# The actual regression test
# ---------------------------------------------------------------------------


class TestGoldenBrainExtraction:
    def test_labels_match_golden(self, golden, golden_prediction) -> None:
        data, _ = golden
        actual = np.asarray(golden_prediction.dataobj).astype(np.uint8)
        golden_labels = np.unpackbits(data["labels_packed"])[: actual.size].astype(
            np.uint8
        )
        _compare_label_masks(actual, golden_labels.reshape(actual.shape))

    def test_probabilities_match_golden(
        self, golden, golden_model, golden_volume
    ) -> None:
        data, meta = golden
        vol, _ = golden_volume

        set_determinism(seed=SEED)
        golden_model.eval()
        prob_result = predict(
            inputs=vol,
            model=golden_model,
            block_shape=tuple(meta["block_shape"]),
            batch_size=meta["batch_size"],
            device="cpu",
            return_labels=False,
        )
        probs_full = np.asarray(prob_result.dataobj).astype(np.float32)  # (C, D, H, W)
        idx = _probe_indices(tuple(meta["volume_shape"]), meta["n_probe_voxels"])
        actual_probe = torch.from_numpy(probs_full[1].ravel()[idx])
        golden_probe = torch.from_numpy(data["probe_probs"])

        # rtol=1e-4 / atol=1e-5: same-device conv3d forward passes reproduce
        # to ~1e-6 across torch patch releases and BLAS backends in practice
        # (verified: regenerating this fixture twice on the same platform
        # gives max_probe_delta == 0.0). This tolerance sits an order of
        # magnitude above that observed noise floor and an order below what
        # a real behavioural change (a re-ordered op, a changed
        # normalization, different weights) would produce -- loose enough to
        # survive a CPU microarchitecture difference, tight enough to catch
        # a real regression. Never the bare torch.testing.assert_close
        # default, which is dtype-derived and undocumented in-place.
        torch.testing.assert_close(actual_probe, golden_probe, rtol=1e-4, atol=1e-5)

    def test_golden_fixture_is_not_degenerate(self, golden) -> None:
        data, meta = golden
        n_total = int(np.prod(meta["volume_shape"]))
        labels = np.unpackbits(data["labels_packed"])[:n_total].astype(np.float32)
        positive_fraction = float(labels.mean())

        # Guards the golden fixture itself, not predict(): an untrained or
        # badly-regenerated model can emit an all-zeros or all-ones mask,
        # against which the sha256/dice/assert_close checks above would all
        # pass vacuously (a degenerate mask matches a degenerate mask).
        assert 0.05 < positive_fraction < 0.95, (
            f"golden positive_fraction={positive_fraction:.4f} looks degenerate; "
            "the fixture itself may need regenerating, not just this test."
        )

    def test_output_contract(self, golden_prediction, golden_volume) -> None:
        vol, _ = golden_volume
        assert isinstance(golden_prediction, nib.Nifti1Image)
        assert golden_prediction.shape == vol.shape
        arr = np.asarray(golden_prediction.dataobj)
        assert arr.dtype == np.float32
        assert set(np.unique(arr).tolist()) <= {0.0, 1.0}
        # predict() constructs a fresh Nifti1Image from np.eye(4) for raw
        # ndarray input (prediction.py:281) -- this pins that contract too.
        assert np.array_equal(golden_prediction.affine, np.eye(4))

    def test_padding_is_exercised(self, golden) -> None:
        _, meta = golden
        volume_shape = tuple(meta["volume_shape"])
        block_shape = tuple(meta["block_shape"])
        non_multiple_axes = sum(
            1 for v, b in zip(volume_shape, block_shape) if v % b != 0
        )
        # Asserts the fixture's own shape choice, so a future regeneration
        # can't silently drop _pad_to_multiple coverage by picking a
        # volume_shape that happens to be an exact multiple of block_shape.
        assert non_multiple_axes >= 2, (
            f"volume_shape={volume_shape} is a multiple of block_shape={block_shape} "
            "on too many axes -- _pad_to_multiple would not be exercised."
        )

    def test_batch_size_does_not_change_result(
        self, golden, golden_model, golden_volume
    ) -> None:
        _, meta = golden
        vol, _ = golden_volume

        set_determinism(seed=SEED)
        golden_model.eval()
        result_a = predict(
            inputs=vol,
            model=golden_model,
            block_shape=tuple(meta["block_shape"]),
            batch_size=2,
            device="cpu",
            return_labels=True,
        )
        set_determinism(seed=SEED)
        golden_model.eval()
        result_b = predict(
            inputs=vol,
            model=golden_model,
            block_shape=tuple(meta["block_shape"]),
            batch_size=8,
            device="cpu",
            return_labels=True,
        )
        # The default (non-strided) path's reassembly is order-independent
        # per-block assignment (prediction.py:60-79), unlike the strided
        # path's accumulation -- so batch_size must not change the result at
        # all. Exact equality, no tolerance: this is the property that makes
        # the sha256 fast path valid in the first place.
        assert np.array_equal(
            np.asarray(result_a.dataobj), np.asarray(result_b.dataobj)
        )

    def test_fixture_meta_is_complete(self, golden) -> None:
        _, meta = golden
        required_keys = {
            "seed",
            "volume_shape",
            "sphere_radius",
            "block_shape",
            "batch_size",
            "arch_kwargs",
            "n_train_steps",
            "n_probe_voxels",
            "torch_version",
            "monai_version",
            "numpy_version",
            "platform",
            "python_version",
        }
        missing = required_keys - meta.keys()
        assert not missing, f"golden fixture meta is missing keys: {missing}"
