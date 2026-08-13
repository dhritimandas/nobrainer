"""Golden-output regression test for ``kwyk_meshnet`` via ``predict()``.

Pins the deterministic (``mc=False``) inference output of a tiny, hermetic
kwyk_meshnet against a committed fixture (``golden/golden_kwyk_meshnet.npz``),
mirroring ``test_golden_outputs.py`` (brain-extraction UNet). This is the
architecture used by the kwyk MONAI bundle
(``scripts/kwyk_reproduction/08_package_bundle.py``); a refactor to
``predict()``'s block handling or to ``KWYKMeshNet.forward``'s ``mc=False``
path could silently change what a bundle consumer gets.

Does NOT use the real converted kwyk weights (a pretrained checkpoint,
excluded from commits by this repo's conventions) -- see
``golden/generate_golden_kwyk.py`` for why a hermetic tiny model is used
instead, same as the UNet fixture.

No network access, no GPU, no downloads. See ``golden/README.md`` for the
full rationale and how to deliberately regenerate the fixture.
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

FIXTURE_PATH = Path(__file__).resolve().parent / "golden" / "golden_kwyk_meshnet.npz"
SEED = 42


# ---------------------------------------------------------------------------
# Helpers duplicated (not imported) from golden/generate_golden_kwyk.py.
# See test_golden_outputs.py's identical comment: nobrainer/sr-tests is not
# a dotted-importable package (hyphen in the name).
# ---------------------------------------------------------------------------


def _make_sphere_volume(
    shape: tuple[int, int, int], radius: float
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]].astype(np.float32)
    center = np.array(shape, dtype=np.float32) / 2
    dist = np.sqrt(sum((coords[i] - center[i]) ** 2 for i in range(3)))
    label = np.zeros(shape, dtype=np.float32)
    label[dist < radius] = 1.0
    label[dist < radius * 0.5] = 2.0
    vol = np.clip(1.0 - dist / dist.max(), 0.0, 1.0).astype(np.float32) * 0.3
    vol = (
        vol
        + (label > 0).astype(np.float32) * 0.4
        + (label == 2).astype(np.float32) * 0.3
    )
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

    Same rationale as test_golden_outputs.py's copy: the default
    (non-strided) prediction path is pure block assignment with no float
    accumulation (nobrainer/prediction.py), so on a fixed platform +
    device="cpu" the label mask is genuinely bit-stable.
    """
    actual_flat = actual.ravel().astype(np.uint8)
    golden_flat = golden.ravel().astype(np.uint8)

    actual_sha = _sha256_of_array(np.packbits(actual_flat))
    golden_sha = _sha256_of_array(np.packbits(golden_flat))
    if actual_sha == golden_sha:
        return

    total = golden_flat.size
    n_diff = int(np.sum(actual_flat != golden_flat))

    # This fixture's non-background region is ~17.5% of 107,520 voxels
    # (measured at generation time -- see golden/README.md); 0.05% of the
    # volume (53 voxels) is the same headroom test_golden_outputs.py uses
    # for its sparser (~6.4%) sphere, generous for last-ULP boundary noise
    # while still catching a real structural change (shifted block, off-by
    # -one pad, flipped axis), which moves voxels throughout the volume.
    max_allowed_diff = int(0.0005 * total)
    assert n_diff <= max_allowed_diff, (
        f"{context}sha256 mismatch AND {n_diff}/{total} label voxels differ "
        f"(budget {max_allowed_diff}). "
        "See nobrainer/sr-tests/golden/README.md before regenerating."
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
    model = get_model("kwyk_meshnet")(**arch)
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
    set_determinism(seed=SEED)
    golden_model.eval()
    # predict() detects kwyk_meshnet's `mc` support (nobrainer/models/_utils.py
    # model_supports_mc, via forward()'s signature) and calls it with
    # mc=False -- the deterministic mean-weights path this fixture pins.
    return predict(
        inputs=vol,
        model=golden_model,
        block_shape=tuple(meta["block_shape"]),
        batch_size=meta["batch_size"],
        device="cpu",
        return_labels=True,
    )


# ---------------------------------------------------------------------------
# The actual regression test
# ---------------------------------------------------------------------------


class TestGoldenKwykMeshnet:
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
        probs_full = np.asarray(prob_result.dataobj).astype(np.float32)
        idx = _probe_indices(tuple(meta["volume_shape"]), meta["n_probe_voxels"])
        actual_probe = torch.from_numpy(probs_full[1].ravel()[idx])
        golden_probe = torch.from_numpy(data["probe_probs"])

        # Same order-of-magnitude rationale as test_golden_outputs.py: an
        # order above the observed same-platform noise floor, an order
        # below a real behavioural change.
        torch.testing.assert_close(actual_probe, golden_probe, rtol=1e-4, atol=1e-5)

    def test_golden_fixture_is_not_degenerate(self, golden) -> None:
        data, meta = golden
        n_total = int(np.prod(meta["volume_shape"]))
        labels = np.unpackbits(data["labels_packed"])[:n_total].astype(np.float32)
        non_bg_fraction = float((labels > 0).mean())
        assert 0.05 < non_bg_fraction < 0.95, (
            f"golden non_bg_fraction={non_bg_fraction:.4f} looks degenerate; "
            "the fixture itself may need regenerating, not just this test."
        )

    def test_output_contract(self, golden_prediction, golden_volume) -> None:
        vol, _ = golden_volume
        assert isinstance(golden_prediction, nib.Nifti1Image)
        assert golden_prediction.shape == vol.shape
        arr = np.asarray(golden_prediction.dataobj)
        assert arr.dtype == np.float32
        # 3 classes (background, shell, core) -- unlike the binary UNet fixture.
        assert set(np.unique(arr).tolist()) <= {0.0, 1.0, 2.0}
        assert np.array_equal(golden_prediction.affine, np.eye(4))

    def test_padding_is_exercised(self, golden) -> None:
        _, meta = golden
        volume_shape = tuple(meta["volume_shape"])
        block_shape = tuple(meta["block_shape"])
        non_multiple_axes = sum(
            1 for v, b in zip(volume_shape, block_shape) if v % b != 0
        )
        assert non_multiple_axes >= 2, (
            f"volume_shape={volume_shape} is a multiple of block_shape={block_shape} "
            "on too many axes -- _pad_to_multiple would not be exercised."
        )

    def test_deterministic_path_is_reproducible(
        self, golden, golden_model, golden_volume
    ) -> None:
        """mc=False must give identical output across repeated calls.

        Not strided/batch-size sensitivity (test_golden_outputs.py covers
        that generically) -- specific to kwyk_meshnet: its default
        forward() (mc=None) samples the VWN weight distribution, so this
        pins that predict()'s explicit mc=False path is what actually runs
        and is deterministic, the property the bundle's determinism caveat
        (docs/README.md, appended by 09_annotate_bundle_readme.py) depends
        on being true.
        """
        _, meta = golden
        vol, _ = golden_volume

        set_determinism(seed=SEED)
        golden_model.eval()
        result_a = predict(
            inputs=vol,
            model=golden_model,
            block_shape=tuple(meta["block_shape"]),
            batch_size=meta["batch_size"],
            device="cpu",
            return_labels=True,
        )
        set_determinism(seed=SEED)
        golden_model.eval()
        result_b = predict(
            inputs=vol,
            model=golden_model,
            block_shape=tuple(meta["block_shape"]),
            batch_size=meta["batch_size"],
            device="cpu",
            return_labels=True,
        )
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
