# Golden-output regression fixture

`golden_brain_extraction.npz` pins the inference output of a tiny, hermetic
brain-extraction UNet against `nobrainer.prediction.predict()`, so a refactor
to padding, block splitting, reassembly, or the argmax/threshold logic cannot
silently change what users get. The test that reads it is
`nobrainer/sr-tests/test_golden_outputs.py`.

## What's in the fixture

- **Weights** (`weights__*` keys) — the state_dict of a `unet(n_classes=2,
  in_channels=1, channels=(4, 8), strides=(2,))`, briefly overfit on a
  synthetic sphere. Includes BatchNorm running stats, which are load-bearing
  for eval-mode reproducibility.
- **Labels** (`labels_packed`, `labels_sha256`) — the full predicted mask
  (`np.packbits`-packed, ~13 KB), plus its digest for the fast-path check.
- **Probability probe** (`probe_probs`) — the class-1 softmax at 4,096
  stride-sampled voxels, for a tolerance-based numeric check independent of
  the exact-match label check.
- **`meta`** (JSON) — seed, volume shape, sphere radius, block shape, batch
  size, architecture kwargs, and the torch/monai/numpy/platform/python
  versions the fixture was generated on. This is what makes a failure
  diagnosable: the assertion output can show *"golden generated with torch
  2.13.0 on darwin-arm64; you are on torch 2.14.0"*, which usually explains
  drift immediately.

The input volume and label are **not stored**. They're regenerated
analytically from `meta` (a deterministic sphere via `np.mgrid`, no RNG at
all) by both `generate_golden.py` and `test_golden_outputs.py` — this keeps
the fixture small and avoids any dependency on RNG-stream stability across
numpy versions.

Total size: ~26 KB, well under the 500 KB hard limit enforced by
`check-added-large-files` in `.pre-commit-config.yaml` (no `args:` override,
so the hook's default `--maxkb=500` applies) and under the 1 MB target.

## Why this test, this way

- **No network, no GPU.** The model is tiny, the volume is synthetic, the
  weights are committed. This runs on every push, on `nobrainer/sr-tests/`'s
  existing CPU CI step, in a few seconds.
- **sha256 fast path, tolerance fallback.** On a fixed platform the label
  mask is genuinely bit-stable (the default, non-strided prediction path is
  pure block assignment with no float accumulation), so an exact digest
  match is the primary signal. A mismatch falls back to a voxel-disagreement
  budget + Dice check, so a legitimate last-ULP boundary difference (e.g. a
  different CPU microarchitecture) doesn't turn into a red build while a
  real regression still fails. See the comments in `_compare_label_masks`
  in `test_golden_outputs.py` for the exact numbers and how they were
  derived from this fixture's actual object size — they are not round
  numbers picked in the abstract.
- **`device="cpu"` is pinned explicitly**, not left to `get_device()`, which
  would return CUDA on the GPU runner and MPS locally. This removes the
  cross-device numerics axis entirely rather than trying to bound it with
  looser tolerances.
- **`monai.utils.set_determinism(seed=42)`** is called before every model
  construction and every `predict()` call, in both scripts. Not strictly
  load-bearing for this architecture (eval mode + `no_grad` + no dropout
  effect already make the forward pass deterministic on a fixed device), but
  cheap and defensive against a future model in this test gaining any stray
  randomness.

## Deliberately regenerating the golden

Regenerating is a reviewable event, not a routine one. Do it only when a
change to `predict()`, the model architecture, or the training procedure
*legitimately* changes the expected output — never to make a failing test
pass without understanding why it failed first.

```bash
# Dry run: builds the same model/volume, prints a diff against the existing
# fixture (voxels differing, Dice, max probe delta, meta diff), does NOT
# write anything.
uv run python nobrainer/sr-tests/golden/generate_golden.py

# Review the printed diff. If it's what you expect, write it:
uv run python nobrainer/sr-tests/golden/generate_golden.py --yes
```

The script refuses to write a degenerate fixture (all-zeros or all-ones
mask, `positive_fraction` outside `(0.05, 0.95)`) — that guard exists because
a degenerate golden would make every downstream assertion pass vacuously.

The commit that updates `golden_brain_extraction.npz` must:
- explain *why* the output legitimately changed, in the commit message
- include the script's printed diff summary (or a description of it) in the
  PR description, since a diff on an opaque binary `.npz` is not reviewable
  on its own
- not bundle unrelated changes

## Future extension

Golden-ing the actual published brain-extraction weights (as opposed to this
hermetic tiny model) would validate the shipped model directly, not just the
`predict()` pipeline. Not implemented here: no weights file is currently
reachable from this repo (no `hf_hub`, `torch.hub`, or pinned download exists
for a brain-extraction checkpoint). If/when one exists, a second test module
following this same sha256-fast-path + tolerance-fallback pattern, gated
behind an opt-in marker or environment variable, is the natural next step.
