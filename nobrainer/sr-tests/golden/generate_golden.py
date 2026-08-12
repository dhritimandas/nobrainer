"""Generate the committed golden fixture for brain-extraction ``predict()`` output.

Run this deliberately to (re)create ``golden_brain_extraction.npz``. It is never
imported or executed by pytest (no ``test_`` prefix). See ``golden/README.md``
for the review discipline expected when regenerating this fixture.

Usage
-----
    uv run python nobrainer/sr-tests/golden/generate_golden.py          # dry run
    uv run python nobrainer/sr-tests/golden/generate_golden.py --yes    # write
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
BLOCK_SHAPE = (16, 16, 16)
BATCH_SIZE = 4
ARCH_KWARGS: dict[str, Any] = {
    "n_classes": 2,
    "in_channels": 1,
    "channels": (4, 8),
    "strides": (2,),
}
N_TRAIN_STEPS = 300
N_PROBE_VOXELS = 4096

FIXTURE_PATH = Path(__file__).resolve().parent / "golden_brain_extraction.npz"


def _make_sphere_volume(
    shape: tuple[int, int, int], radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic volume + binary sphere label.

    No RNG -- pure ``meshgrid`` arithmetic, so it reproduces bit-for-bit across
    numpy versions and platforms, unlike a seeded ``np.random`` draw.
    """
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]].astype(np.float32)
    center = np.array(shape, dtype=np.float32) / 2
    dist = np.sqrt(sum((coords[i] - center[i]) ** 2 for i in range(3)))
    label = (dist < radius).astype(np.float32)
    vol = np.clip(1.0 - dist / dist.max(), 0.0, 1.0).astype(np.float32) * 0.3
    vol = vol + label * 0.7  # sphere brighter than background
    return vol, label


def _probe_indices(volume_shape: tuple[int, int, int], n_probe: int) -> np.ndarray:
    """Fixed stride-sampled flat indices into a flattened volume.

    Deterministic from ``volume_shape``/``n_probe`` alone, so neither script
    needs to store or pass indices explicitly -- both recompute identically.
    """
    total = int(np.prod(volume_shape))
    stride = total // n_probe
    return (np.arange(n_probe) * stride).astype(np.int64)


def _train(vol: np.ndarray, label: np.ndarray) -> nn.Module:
    set_determinism(seed=SEED)
    model = get_model("unet")(**ARCH_KWARGS)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.from_numpy(vol[None, None])  # (1, 1, D, H, W)
    label_long = torch.from_numpy(label).long().unsqueeze(0)  # (1, D, H, W)

    for _ in range(N_TRAIN_STEPS):
        optimizer.zero_grad()
        loss = criterion(model(x), label_long)
        loss.backward()
        optimizer.step()

    return model


def generate() -> dict[str, Any]:
    """Build, train, and predict -- returns everything the fixture needs."""
    vol, label = _make_sphere_volume(VOLUME_SHAPE, SPHERE_RADIUS)
    model = _train(vol, label)

    # set_determinism before every run that invokes the model -- defensive,
    # not strictly load-bearing for a plain eval-mode unet (no dropout effect,
    # no MC sampling), but cheap and guards against a future model in this
    # test gaining stray randomness.
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
    foreground_probs = probs_full[1]  # class 1 = brain

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
        "arch_kwargs": {
            "n_classes": ARCH_KWARGS["n_classes"],
            "in_channels": ARCH_KWARGS["in_channels"],
            "channels": list(ARCH_KWARGS["channels"]),
            "strides": list(ARCH_KWARGS["strides"]),
        },
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
    intersection = float((old_labels * new_labels).sum())
    dice = 2 * intersection / (old_labels.sum() + new_labels.sum() + 1e-8)
    max_probe_delta = float(np.max(np.abs(old["probe_probs"] - new["probe_probs"])))

    print("\n--- Diff vs existing golden fixture ---")
    print(f"  label voxels differing: {n_diff}/{total} ({100 * n_diff / total:.4f}%)")
    print(f"  dice (old vs new): {dice:.6f}")
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

    print(f"Generating golden fixture (seed={SEED}, volume_shape={VOLUME_SHAPE}) ...")
    new_data = generate()

    existing = _load_existing()
    if existing is not None:
        _print_diff(existing[0], existing[1], new_data)
    else:
        print("No existing fixture -- this will create a new one.")

    positive_fraction = float(new_data["labels_full"].mean())
    if not (0.05 < positive_fraction < 0.95):
        raise SystemExit(
            f"Refusing to write a degenerate golden: positive_fraction="
            f"{positive_fraction:.4f} is outside (0.05, 0.95). The overfit likely failed."
        )
    print(f"positive_fraction={positive_fraction:.4f} (sanity range 0.05-0.95: OK)")

    if not args.yes:
        print("\nDry run -- pass --yes to overwrite the fixture.")
        return

    _write(new_data)
    size_kb = FIXTURE_PATH.stat().st_size / 1024
    print(f"Wrote {FIXTURE_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
