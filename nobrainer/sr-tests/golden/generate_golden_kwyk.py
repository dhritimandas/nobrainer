"""Generate the committed golden fixture for kwyk_meshnet ``predict()`` output.

Run this deliberately to (re)create ``golden_kwyk_meshnet.npz``. It is never
imported or executed by pytest (no ``test_`` prefix). See ``golden/README.md``
for the review discipline expected when regenerating this fixture.

This mirrors ``generate_golden.py`` (brain-extraction UNet) exactly, but for
the ``kwyk_meshnet`` architecture used by the kwyk MONAI bundle
(``scripts/kwyk_reproduction/08_package_bundle.py``). It does NOT embed the
real converted kwyk weights: those are pretrained model weights, which this
repo's conventions exclude from commits. Instead, like the UNet fixture, it
hermetically trains a tiny kwyk_meshnet from scratch and pins ITS output --
this tests that kwyk_meshnet's deterministic (``mc=False``) inference path
through ``predict()`` stays bit-stable across refactors, independent of any
specific pretrained checkpoint.

Usage
-----
    uv run python nobrainer/sr-tests/golden/generate_golden_kwyk.py          # dry run
    uv run python nobrainer/sr-tests/golden/generate_golden_kwyk.py --yes    # write
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import monai
from monai.utils import set_determinism
import numpy as np
import torch
import torch.nn as nn

from nobrainer.models import get as get_model
from nobrainer.prediction import predict

SEED = 42
# Not a multiple of BLOCK_SHAPE on 2 of 3 axes -- exercises _pad_to_multiple.
VOLUME_SHAPE = (40, 48, 56)
SPHERE_RADIUS = 12.0
# kwyk's real (published) block shape -- the SavedModel's fixed input size --
# not an arbitrary choice like the UNet fixture's 16^3.
BLOCK_SHAPE = (32, 32, 32)
BATCH_SIZE = 4
ARCH_KWARGS: dict[str, Any] = {
    "n_classes": 3,
    "in_channels": 1,
    "filters": 8,
    "receptive_field": 37,
    "dropout_type": "bernoulli",
    "bias": True,
}
N_TRAIN_STEPS = 500
N_PROBE_VOXELS = 4096

FIXTURE_PATH = Path(__file__).resolve().parent / "golden_kwyk_meshnet.npz"


def _make_sphere_volume(
    shape: tuple[int, int, int], radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic volume + 3-class (bg, shell, core) label.

    No RNG -- pure ``meshgrid`` arithmetic, so it reproduces bit-for-bit
    across numpy versions and platforms, unlike a seeded ``np.random`` draw.
    Two concentric spheres (not one, unlike the UNet fixture) so a
    multi-class model has more than one non-background class to learn.
    """
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]].astype(np.float32)
    center = np.array(shape, dtype=np.float32) / 2
    dist = np.sqrt(sum((coords[i] - center[i]) ** 2 for i in range(3)))
    label = np.zeros(shape, dtype=np.float32)
    label[dist < radius] = 1.0  # shell
    label[dist < radius * 0.5] = 2.0  # core
    vol = np.clip(1.0 - dist / dist.max(), 0.0, 1.0).astype(np.float32) * 0.3
    vol = (
        vol
        + (label > 0).astype(np.float32) * 0.4
        + (label == 2).astype(np.float32) * 0.3
    )
    return vol, label


def _probe_indices(volume_shape: tuple[int, int, int], n_probe: int) -> np.ndarray:
    """Fixed stride-sampled flat indices into a flattened volume."""
    total = int(np.prod(volume_shape))
    stride = total // n_probe
    return (np.arange(n_probe) * stride).astype(np.int64)


def _train(vol: np.ndarray, label: np.ndarray) -> nn.Module:
    set_determinism(seed=SEED)
    model = get_model("kwyk_meshnet")(**ARCH_KWARGS)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.from_numpy(vol[None, None])  # (1, 1, D, H, W)
    label_long = torch.from_numpy(label).long().unsqueeze(0)  # (1, D, H, W)

    for _ in range(N_TRAIN_STEPS):
        optimizer.zero_grad()
        # mc=False: the deterministic mean-weights path -- matches how
        # predict() always calls a supports_mc model, and how the real
        # kwyk MAP checkpoint is meant to be used.
        loss = criterion(model(x, mc=False), label_long)
        loss.backward()
        optimizer.step()

    return model


def generate() -> dict[str, Any]:
    """Build, train, and predict -- returns everything the fixture needs."""
    vol, label = _make_sphere_volume(VOLUME_SHAPE, SPHERE_RADIUS)
    model = _train(vol, label)

    set_determinism(seed=SEED)
    model.eval()
    label_result = predict(
        inputs=vol,
        model=model,
        block_shape=BLOCK_SHAPE,
        batch_size=BATCH_SIZE,
        device="cpu",
        return_labels=True,
    )
    labels_full = np.asarray(label_result.dataobj).astype(np.float32)

    set_determinism(seed=SEED)
    prob_result = predict(
        inputs=vol,
        model=model,
        block_shape=BLOCK_SHAPE,
        batch_size=BATCH_SIZE,
        device="cpu",
        return_labels=False,
    )
    # Default (non-strided) path, C > 1: (C, D, H, W) -- see prediction.py.
    probs_full = np.asarray(prob_result.dataobj).astype(np.float32)
    foreground_probs = probs_full[1]  # class 1 = shell

    idx = _probe_indices(VOLUME_SHAPE, N_PROBE_VOXELS)
    probe_probs = foreground_probs.ravel()[idx]

    labels_packed = np.packbits(labels_full.ravel().astype(np.uint8))
    labels_sha256 = hashlib.sha256(labels_packed.tobytes()).hexdigest()

    weights = {
        f"weights__{k}": v.detach().numpy() for k, v in model.state_dict().items()
    }

    meta = {
        "seed": SEED,
        "volume_shape": list(VOLUME_SHAPE),
        "sphere_radius": SPHERE_RADIUS,
        "block_shape": list(BLOCK_SHAPE),
        "batch_size": BATCH_SIZE,
        "arch_kwargs": dict(ARCH_KWARGS),
        "n_train_steps": N_TRAIN_STEPS,
        "n_probe_voxels": N_PROBE_VOXELS,
        "torch_version": torch.__version__,
        "monai_version": monai.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    return {
        "labels_full": labels_full,
        "labels_packed": labels_packed,
        "labels_sha256": labels_sha256,
        "probe_probs": probe_probs,
        "meta": meta,
        "weights": weights,
    }


def _load_existing() -> tuple[np.lib.npyio.NpzFile, dict[str, Any]] | None:
    if not FIXTURE_PATH.exists():
        return None
    old = np.load(FIXTURE_PATH, allow_pickle=False)
    old_meta = json.loads(bytes(old["meta"]).decode("utf-8"))
    return old, old_meta


def _print_diff(
    old: np.lib.npyio.NpzFile, old_meta: dict[str, Any], new: dict[str, Any]
) -> None:
    old_labels = np.unpackbits(old["labels_packed"])[
        : int(np.prod(VOLUME_SHAPE))
    ].astype(np.float32)
    new_labels = new["labels_full"].ravel()
    n_diff = int(np.sum(old_labels != new_labels))
    total = old_labels.size
    max_probe_delta = float(np.max(np.abs(old["probe_probs"] - new["probe_probs"])))

    print("\n--- Diff vs existing golden fixture ---")
    print(f"  label voxels differing: {n_diff}/{total} ({100 * n_diff / total:.4f}%)")
    print(f"  max probe prob delta: {max_probe_delta:.6g}")
    print(f"  old meta: {json.dumps(old_meta, indent=2)}")
    print(f"  new meta: {json.dumps(new['meta'], indent=2)}")
    print("----------------------------------------\n")


def _write(data: dict[str, Any]) -> None:
    payload = {
        "labels_packed": data["labels_packed"],
        "labels_sha256": np.frombuffer(
            data["labels_sha256"].encode("utf-8"), dtype=np.uint8
        ),
        "probe_probs": data["probe_probs"],
        "meta": np.frombuffer(json.dumps(data["meta"]).encode("utf-8"), dtype=np.uint8),
        **data["weights"],
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE_PATH, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually write the fixture (default: dry run).",
    )
    args = parser.parse_args()

    print(
        f"Generating kwyk golden fixture (seed={SEED}, volume_shape={VOLUME_SHAPE}) ..."
    )
    new_data = generate()

    existing = _load_existing()
    if existing is not None:
        _print_diff(existing[0], existing[1], new_data)
    else:
        print("No existing fixture -- this will create a new one.")

    labels = new_data["labels_full"]
    non_bg_fraction = float((labels > 0).mean())
    if not (0.05 < non_bg_fraction < 0.95):
        raise SystemExit(
            f"Refusing to write a degenerate golden: non_bg_fraction="
            f"{non_bg_fraction:.4f} is outside (0.05, 0.95). The overfit likely failed."
        )
    print(f"non_bg_fraction={non_bg_fraction:.4f} (sanity range 0.05-0.95: OK)")

    if not args.yes:
        print("\nDry run -- pass --yes to overwrite the fixture.")
        return

    _write(new_data)
    size_kb = FIXTURE_PATH.stat().st_size / 1024
    print(f"Wrote {FIXTURE_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
