# kwyk parity report: converted PyTorch model vs. the original container

`scripts/kwyk_reproduction/07_parity_kwyk.py` checks whether a PyTorch
checkpoint produced by `nobrainer.datasets.convert_kwyk` reproduces the
original `neuronets/kwyk` TensorFlow container's output on identical input,
end to end (conform -> normalize -> block -> infer -> reassemble). It is
separate from `05_compare_kwyk.py`, which measures Dice against ground
truth for each side independently and does not compare the two models to
each other.

## Why a container-based check, not a pure-numpy one

`mri_convert --conform` (FreeSurfer) does an intensity-histogram-based
`uchar` conversion followed by trilinear reslicing. The `uchar` step is not
a plain min-max scale (verified empirically: a synthetic ramp plus an
outlier does not get squashed by the outlier) and cannot be reproduced in
numpy. The rule this harness follows: **conform exactly once, inside the
container, and feed the identical conformed volume to both sides.** A
volume already at 256^3 is not treated differently by `mri_convert` -- the
kwyk CLI's own skip-guard is shape-only -- so conforming is always safe to
run, even on an already-conformed volume.

## Preprocessing contract

```
raw T1 --[container: mri_convert --conform]--> conformed.nii.gz (256^3, uint8, 0-255)
                                                     |
                        +----------------------------+----------------------------+
                        v                                                         v
             TF SavedModel (in container)                          PyTorch (host)
             z = (a - a.mean()) / a.std()   <-- computed ONCE on the host, fed identically to both sides
```

Normalization is a **global whole-volume z-score** on the conformed
(0-255) values, float32, `ddof=0` -- matching
`/usr/local/lib/python3.5/dist-packages/nobrainer/volume.py:zscore` inside
the container exactly (verified by reading that file inside a live
container). It is computed once on the host and passed to both sides with
`normalizer=None`: `nobrainer.prediction.predict()` normalizes **per
block** when given a `normalizer` callable, which would silently produce
different input than a whole-volume z-score.

`assert_preprocessing()` independently reruns the container's own
`zscore()` on the same conformed array and diffs it against the host
computation before any inference runs. A mismatch aborts as a
preprocessing error, never surfaces later as a misleading Dice number.
The gate is `1e-4`, not `1e-6`: an observed run (arm64 host vs. an
amd64-emulated container, both plain numpy, identical formula, 16.7M
elements) measured `9.5e-06` -- whole-volume float32 reduction is not
associative across BLAS backends, so a small residual here is expected,
not a bug.

## Verified container internals

These were confirmed against a live `docker run neuronets/kwyk:latest-cpu`
container, not assumed:

| What | Where | Value |
|---|---|---|
| SavedModel dirs | `/opt/kwyk/saved_models/<name>/<timestamp>/` | `all_50_wn` (MAP), `all_50_bwn_09_multi` (BD), `all_50_bvwn_multi_prior` (SSD) |
| MAP input/output signature | `saved_model_cli show --dir ... --all` | in: `volume` `(-1,32,32,32,1)` float32; out: `class_ids` int64, `logits` `(-1,32,32,32,50)`, `probabilities` `(-1,32,32,32,50)` |
| `mri_convert` | `/opt/kwyk/freesurfer/bin/mri_convert` | FreeSurfer `stable6` |
| TF / Python | container | TensorFlow 1.12.3, Python 3.5.2 |
| Blocking order | `nobrainer/volume.py:to_blocks` (container) | reshape to `(nd,bd,nh,bh,nw,bw)`, `transpose(0,2,4,1,3,5)` -- identical to `nobrainer.prediction._extract_blocks` on the PyTorch side |
| Stage B variance | `nobrainer/predict.py:predict_from_array` | `np.sum(M / n_samples, axis=-1)` -- **sum** over classes |
| Stage B entropy | same | `-np.sum(log(p + 1e-7) * p, axis=-1)` |
| PyTorch variance (`predict_with_uncertainty`) | `nobrainer/prediction.py` | **mean** over classes, `eps=1e-8` -- reconciled via `parity_metrics.reconcile_pytorch_variance` (`* n_classes`); the entropy epsilon difference (1e-7 vs 1e-8) is documented, not "fixed" |

Extraction of TF variables to an `.npz` for `convert_kwyk.py --npz` uses
`tf.train.load_checkpoint` pointed at
`<saved_model_dir>/variables/variables` (the SavedModel directory itself
is not a valid checkpoint prefix).

## Stages

- **Stage A** (`all_50_wn`, the MAP model): the only one of the three
  SavedModels observed to be run-to-run deterministic (`max|run0-run1|
  = 0.0` across two runs; the BD and SSD models are not, even with the CLI's
  own `-n 1`). Compares raw logits directly -- `nobrainer.prediction.predict()`
  only exposes softmax probabilities or argmax labels, so Stage A reuses
  that module's private block-extraction/stitching helpers
  (`_pad_to_multiple`, `_extract_blocks`, `_stitch_blocks` -- identical math
  to the public API) and calls the model directly to keep pre-softmax
  values.
- **Stage B** (`all_50_bvwn_multi_prior`, the SSD model): MC/uncertainty,
  aggregate statistics only, never per-sample -- the two frameworks' RNGs
  are unrelated, and PyTorch's `ConcreteDropout3d` draws one mask shared
  across the whole batch, so its sample statistics depend on `batch_size`.
  Runs on a central crop (`--crop-size-b`, default 128) with
  `--n-samples-b` (default 20) samples: full-volume MC at kwyk's block
  rate is hours, not minutes, and aggregate statistics do not need
  full-volume coverage. The container side reuses the container's own MC
  loop verbatim (`nobrainer.predict.predict_from_array`) rather than
  reimplementing it.

Neither stage uses ground-truth segmentation labels: this is a
model-to-model parity check, not an accuracy check (that is `05`'s job).
Stage A's `ece_pytorch`/`ece_tf` use the models' mutual agreement mask as
the "correct" signal for each side's own confidence -- there is no ground
truth to calibrate against in a pure parity context.

## Running it

```bash
# preprocessing guard alone (fast: one conform + one zscore, no inference)
uv run python scripts/kwyk_reproduction/07_parity_kwyk.py \
  --volume sub-01_t1.mgz --check-preprocessing-only --work-dir results/parity

# Stage A on one volume
uv run python scripts/kwyk_reproduction/07_parity_kwyk.py \
  --volume sub-01_t1.mgz --pytorch-weights-map kwyk_map.pth \
  --stage a --out results/parity/parity_report.json

# both stages
uv run python scripts/kwyk_reproduction/07_parity_kwyk.py \
  --volume sub-01_t1.mgz \
  --pytorch-weights-map kwyk_map.pth --pytorch-weights-ssd kwyk_ssd.pth \
  --stage both --out results/parity/parity_report.json
```

`--pytorch-weights-map`/`--pytorch-weights-ssd` are produced by
`nobrainer.datasets.convert_kwyk` from an `.npz` extracted the same way
(`tf.train.load_checkpoint` against `<saved_model_dir>/variables/variables`
inside a `neuronets/kwyk` container).

## Observed run (sub-01, this repo's dev machine)

Real numbers from an actual run, not fabricated -- see
`results/parity/parity_report.json` (not committed; a run artifact).

- **Preprocessing**: `input_max_abs_diff = 9.5e-06`, matched.
- **Stage A** (64^3 crop, 8 of 512 blocks -- see Hardware requirements
  below for why not full-volume): `max_abs_logit_diff = 6.9e-05`,
  `dice_mean = 1.0`, `voxel_agreement = 1.0`. All gates pass comfortably.
- **Stage B** (full designed scope: 128^3 crop, N=20 MC samples; ~83 min
  wall clock under QEMU emulation -- ~68 min TF, ~14 min PyTorch):
  `mean_label_dice = 0.9804` and `entropy_pearson_r = 0.9963` pass their
  gates comfortably. **`variance_pearson_r = 0.8740` misses the 0.90
  gate** (`variance_scale_ratio = 0.816`: PyTorch's reconciled variance
  runs ~18% below TF's in this crop).

  Plausible cause, not confirmed: MC sample variance is a much noisier
  statistic than the sample mean at N=20, and
  `ConcreteDropout3d` (`vwn_layers.py`) draws **one dropout mask shared
  across the whole batch**, not independent per-sample masks -- a
  documented implementation difference (see the module's docstring) that
  could suppress inter-sample variance relative to TF's per-sample
  sampling. Increasing `--n-samples-b` tightens the variance estimate on
  both sides and is the natural next check before concluding this is a
  real discrepancy rather than sampling noise at N=20.

## Hardware requirements

Stage A holds a full `(256,256,256,50)` float32 array (`~3.35GB`) on both
the container and host sides; Docker's VM needs at least ~8GB, and host
RAM should have headroom well above that once PyTorch's own tensors and
the TF graph are counted. The `neuronets/kwyk` image is `linux/amd64` only
-- on an arm64 host (e.g. Apple Silicon) it runs under QEMU emulation,
which is measurably slow for TF1.12 SavedModel inference (minutes for a
single graph load, well beyond native speed). A full 512-block Stage A run
under emulation should be expected to take a long time; budget accordingly
or run on amd64 hardware / a cloud instance.

## Reading the JSON

Every threshold applied is echoed into `thresholds`, so a report is
self-describing. `result.pass` is the overall verdict; `result.failed_checks`
names which stage/volume combinations failed. `preprocessing` records the
per-volume conform/normalize verification (`input_match` must be `true`
for that volume's Stage A/B results to be meaningful). See the plan at
`.claude/plans/kwyk_parity_PLAN.md` for the full JSON schema and the
per-metric threshold justifications.

## Debugging a failure

If Stage A fails *and* `preprocessing.input_match` is `true` for that
volume, the mismatch is in the pipeline (blocking, reassembly, dtype), not
the weight mapping -- `offline_parity_check`
(`nobrainer/datasets/convert_kwyk.py`) already gates the mapping itself at
conversion time. If conversion's own parity check failed, the debug order
is axis permutation -> `g` shape branch -> layer index base (not bias,
which that check cannot observe by construction).
