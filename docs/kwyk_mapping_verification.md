# kwyk TF → PyTorch variable mapping: verification report

**Status:** verification report (no code was changed as part of producing it).
**Update:** all six discrepancies have since been fixed and re-validated against
the real weights and the live TF graph — see §6 (Resolutions).

**Verified against:** `neuronets/kwyk:latest-cpu`, digest
`sha256:8b72179a0b99284c5a520dde61982030891e6461946a5c36e67a445dd59226e1`,
SavedModels at `/opt/kwyk/saved_models/`:

| model | timestamp | variables |
|---|---|---|
| `all_50_wn` (MAP) | 1555341859 | 41 (40 weights + `global_step`) |
| `all_50_bwn_09_multi` (BD) | 1555963478 | 41 (40 weights + `global_step`) |
| `all_50_bvwn_multi_prior` (SSD) | 1556816070 | 48 (47 weights + `global_step`) |

**Method.** Three independent agents inventoried the PyTorch side, the TF side, and
the converter's stated assumptions, without reading each other's targets. Their
inventories were then cross-checked, and — beyond what any static inventory can
settle — the mapping was validated **numerically against the live TF graph**
(§4).

Converter under review: `nobrainer/datasets/convert_kwyk.py`.
PyTorch target: `nobrainer/models/bayesian/{vwn_layers,kwyk_meshnet}.py`.

---

## 1. Mapping table

Published configuration: `filters=96`, `receptive_field=37`, `n_classes=50`,
`in_channels=1`. TF names carry **no** trailing `:0` as returned by
`NewCheckpointReader`. PyTorch indices are 0-based, TF indices are 1-based;
`torch_i = tf_i - 1`.

### Hidden layers (TF `layer_1..layer_7` → PyTorch `layer_0..layer_6`)

| TF variable | TF shape | PyTorch key | PT shape | Transform |
|---|---|---|---|---|
| `layer_{i}/conv3d/v` | `[3,3,3,1,96]` (i=1)<br>`[3,3,3,96,96]` (i=2..7) | `layer_{i-1}.conv.v` | `(96,1,3,3,3)`<br>`(96,96,3,3,3)` | `transpose(4,3,0,1,2)`, `→float32` |
| `layer_{i}/conv3d/g` | `[1,1,1,1,96]` | `layer_{i-1}.conv.g` | `(96,1,1,1,1)` | `transpose(4,3,0,1,2)` |
| `layer_{i}/conv3d/kernel_a` | `[3,3,3,1,96]` / `[3,3,3,96,96]` | `layer_{i-1}.conv.kernel_a` | `(96,1,3,3,3)` / `(96,96,3,3,3)` | `transpose(4,3,0,1,2)` |
| `layer_{i}/conv3d/bias_m` | `[96]` | `layer_{i-1}.conv.bias_m` | `(96,)` | verbatim copy — **only if `create_bias=True`** |
| `layer_{i}/conv3d/bias_a` | `[96]` | `layer_{i-1}.conv.bias_a` | `(96,)` | verbatim copy — **only if `create_bias=True`** |
| `layer_{i}/concrete_dropout/p` | `[96]` (SSD only) | `layer_{i-1}.dropout.p_logit` | `(96,)` | `clip(p, 0.05, 0.95)` → `log(p/(1−p))` |

### Output layer (TF `logits/` → PyTorch `classifier`)

| TF variable | TF shape | PyTorch key | PT shape | Transform |
|---|---|---|---|---|
| `logits/conv3d/v` | `[1,1,1,96,50]` | `classifier.weight` | `(50,96,1,1,1)` | `transpose(4,3,0,1,2)` then weight-norm collapse `g·v/‖v‖` |
| `logits/conv3d/g` | `[1,1,1,1,50]` | *(folded into `classifier.weight`)* | — | multiplied into the collapse above |
| `logits/conv3d/bias_m` | `[50]` | `classifier.bias` | `(50,)` | verbatim copy |
| `logits/conv3d/kernel_a` | `[1,1,1,96,50]` | **— none —** | — | **discarded** (logged) |
| `logits/conv3d/bias_a` | `[50]` | **— none —** | — | **discarded silently** (never read) |

### Not mapped

| TF variable | Reason |
|---|---|
| `global_step` (scalar, `int64`) | Training bookkeeping, not a weight. Ignored by the `^layer_(\d+)/` regex and by all templates. |

---

## 2. CONFIRM / REFUTE

### (a) TF conv filters are `[k,k,k,in,out]` and need `transpose(4,3,0,1,2)` — **CONFIRMED**

Two independent lines of evidence.

*Structural.* `layer_1/conv3d/v` has shape `[3,3,3,1,96]`. Layer 1 is the only
layer with `in_channels=1`, and the `1` sits at **axis 3** with `96` at **axis 4** —
so axis 3 is `in`, axis 4 is `out`, and axes 0–2 are the spatial kernel. A middle
layer is `[3,3,3,96,96]`, consistent. PyTorch requires `(out,in,k,k,k)` =
`(96,1,3,3,3)` (`vwn_layers.py:98`). The permutation `(4,3,0,1,2)` sends
axis4→0, axis3→1, axes0,1,2→2,3,4 — exactly correct.

*Numerical.* See §4: converted weights reproduce the real TF graph's logits to
`max|Δ| = 9.92e-05`. A wrong permutation could not produce that.

### (b) TF `g` is `[1,1,1,1,out]` or `[out]` — **CONFIRMED (5-D only); the 1-D branch is never exercised**

Every `g` in all three models is **5-D**: `[1,1,1,1,96]` for hidden layers,
`[1,1,1,1,50]` for logits. A 1-D `[out]` `g` does **not occur** in any published
checkpoint.

The converter's disjunction is *permissive*, so it is not refuted — but the 1-D
branch (`convert_kwyk.py:199-204`) is dead code with respect to real data,
reachable only from the synthetic unit-test fixture. Not a bug; recorded so it is
not mistaken for validated behaviour. See Discrepancy **D4**.

### (c) TF layer index base, and whether it matches `_n_layers` — **CONFIRMED: base = 1, span matches**

TF hidden layers span `layer_1 … layer_7` — **1-based**, 7 distinct indices, no
`layer_0`, no `layer_8`. PyTorch `_n_layers = 7` for *all three* receptive fields
(`_constants.py:7-11`; each schedule has 7 entries), and layers are attributes
`layer_0 … layer_6` (`kwyk_meshnet.py:190`).

`detect_layer_index_base` computes `span = 7 - 1 + 1 = 7`, requires
`span == n_layers` → `7 == 7` ✓, and requires `lo ∈ (0,1)` → `1` ✓. It returns
`1`, so `tf_i = torch_i + 1` maps `layer_0..layer_6` ↔ `layer_1..layer_7`
correctly.

Critically, this equality **only holds because the output layer is in its own
`logits/` namespace** rather than being `layer_8`. Had it been `layer_8`, span
would be 8 and the check would reject every real checkpoint.

Caveat on the check's strength: it compares a *span*, not a set, so a checkpoint
with a gap (e.g. indices {1,2,4,5,6,7,8}) would pass. Not a defect for these
checkpoints — noted for completeness.

### (d) `bias_m` and `bias_a` present in TF for every layer — **CONFIRMED, for all 8 conv layers in all 3 models (24/24)**

Both are present for every hidden layer *and* for the logits layer, in all three
models. Shapes `[96]` hidden, `[50]` logits.

This confirmation is what makes **D1** and **D2** serious rather than theoretical:
the converter defaults to *discarding* data that is always present.

### (e) Concrete dropout `p` is stored as a probability, not a logit — **CONFIRMED**

All 672 values (7 layers × 96) lie in `[0.607895, 0.954055]` — strictly inside
`(0,1)`, no negatives, none above 1. These are probabilities. PyTorch stores a
**logit** (`vwn_layers.py:195-197`), so a transform is genuinely required, and the
converter's `log(p/(1−p))` is the correct direction.

`p` exists for hidden layers 1–7 only (7 variables); the logits layer has none,
matching PyTorch, where only `_VWNLayerConcrete` carries a `ConcreteDropout3d`.

The *direction* is right. The *clamp* applied alongside it is not lossless — see
**D3**.

---

## 3. DISCREPANCIES

Ordered by severity. Each names the exact line that would have to change.

### D1 — Default conversion silently discards biases that are always present (**HIGH**)

`KWYKMeshNet` defaults to `bias=False` (`kwyk_meshnet.py:157`), and `--create-bias`
defaults to off (`convert_kwyk.py:471-472`). But §2(d) establishes every real
checkpoint has `bias_m`/`bias_a` for every layer. So the **default invocation
produces a numerically wrong model** and only emits a `logger.warning`
(`convert_kwyk.py:283-289`).

Measured cost, MAP model vs the real TF graph on identical input:

| run | max&#124;Δ logits&#124; | mean&#124;Δ&#124; |
|---|---|---|
| `--create-bias` (biases kept) | **9.92e-05** | 1.15e-05 |
| default (biases dropped) | **19.77** | 2.66 |

That is a ~200,000× degradation, reported at WARNING level while the command exits
0. Worse, the offline parity check **passes anyway** in the no-bias case, because
it compares `conv.bias_m` against itself (`convert_kwyk.py:336` uses the model's
own bias on both sides) — so the built-in gate does not catch this.

*Also note the module docstring at `kwyk_meshnet.py:138-143` asserts the published
checkpoints are bias-free. That statement is **false** and should be corrected.*

**Lines to change:** `convert_kwyk.py:283-289` — make the bias-drop an error, or
invert the default so biases are kept unless explicitly discarded. Consequential
edit at `convert_kwyk.py:471-472` (flag default) and `:529` (model construction).

### D2 — The logits layer's variational parameters are unrepresentable, so MC inference is not faithful (**HIGH, architectural**)

The TF output layer is a **full VWN conv**: `logits/conv3d/{v,g,kernel_a,bias_m,bias_a}`.
`KWYKMeshNet.classifier` is a plain `nn.Conv3d` (`kwyk_meshnet.py:192`), which can
represent only the mean path. So `kernel_a` (logged, `convert_kwyk.py:338-342`) and
`bias_a` (**never read — there is no `_TF_LOGITS_BIAS_A` constant at all**,
`convert_kwyk.py:81-84`) are dropped.

Consequence: for **deterministic** inference this is harmless and parity is
excellent (§4). For **MC/uncertainty** inference — the entire point of the SSD
model — the converted model's output layer is deterministic while the original's
samples. Measured on the real graph, the TF SSD model varies by
**max|Δ| = 11.52** between two runs on identical input, and TF BD by **39.93**;
the converted PyTorch classifier contributes **zero** of that variance.

Uncertainty estimates from the converted SSD model will therefore be
systematically under-dispersed at the output layer. That is a real fidelity limit,
not a rounding issue, and it should be documented wherever the converted weights
are published.

**Lines to change:** either `kwyk_meshnet.py:192` (make the classifier an
`FFGConv3d` so the variational output layer is representable) or, if the
limitation is accepted, `convert_kwyk.py:338-342` should be raised from
`logger.info` to `logger.warning` and state the MC consequence explicitly.

### D3 — 89% of concrete-dropout probabilities are silently clamped (**MEDIUM**)

`_p_to_logit` clips to `[0.05, 0.95]` (`convert_kwyk.py:218`, constants at `:57-58`)
before taking the logit. But the real `p` values run up to **0.954055**, and
**598 of 672 (89.0%)** exceed 0.95. Layer 7 is worst: its minimum is exactly
0.950000, so **95 of its 96 values** are clamped.

Maximum absolute distortion is small (0.004055 in probability space), and the
clamp does mirror `ConcreteDropout3d.p`'s own `.clamp(0.05, 0.95)`
(`vwn_layers.py:202`) — so the converter is *faithful to the PyTorch forward*.
The discrepancy is that **PyTorch's clamp range is narrower than the range TF
actually trained to**, so the PyTorch model cannot express the trained dropout
rates regardless of the converter. The clustering just under 0.954 and the exact
0.950000 floor suggest TF applied its own constraint at a slightly wider bound.

Because `p` scales activations directly (`vwn_layers.py:218`, `x * p`), a
systematic ~0.4% downward bias on 89% of channels is a small but real, one-signed
error — not noise that cancels.

**Lines to change:** `convert_kwyk.py:57-58` (the clamp constants) *and*
`vwn_layers.py:202` (the forward clamp) must move together — changing only the
converter would be undone at forward time. Alternatively, keep the clamp but log
how many values were affected, at `convert_kwyk.py:218`.

### D4 — The 1-D `g` branch is unreachable for real checkpoints (**LOW, informational**)

`_g_tf_to_torch` accepts 1-D `g` (`convert_kwyk.py:199-204`), but every real `g` is
5-D (§2b). The branch is exercised only by the synthetic test fixture, so it is
untested against reality and could mask a genuine shape anomaly by silently
`reshape`-ing it.

**Lines to change:** none required. If tightening is desired,
`convert_kwyk.py:199-204` could log when the 1-D path is taken.

### D5 — MAP and BD checkpoints are structurally indistinguishable (**LOW, operational**)

`all_50_wn` and `all_50_bwn_09_multi` have byte-identical key sets, shapes, and
dtypes — they differ only in values and in whether MC is enabled at inference.
The converter cannot tell them apart, and `dropout_type` / `dropout_rate` /
MC-at-inference are **not** stored in the PyTorch `state_dict` either
(`nn.Dropout3d` is parameterless). A user who converts BD weights and runs them
deterministically gets the MAP behaviour with no warning from either side.

**Lines to change:** none in the mapping. Provenance (source model name) should be
recorded alongside the output `.pth`; today `main()` (`convert_kwyk.py:508-545`)
writes a bare `state_dict` with no metadata.

### D6 — Stale prose in the converter's own docstring (**LOW**)

`convert_kwyk.py:11-13` claims the axis conventions were *"verified against …
ARCHITECTURE.md and the source snippets of `vwn_layers.py`"*. They are now verified
against the live checkpoint and the live TF graph, which is materially stronger.
Separately, `ARCHITECTURE.md:115` describes the output as *"Layer 8 (logits)"*,
which reads as though it were `layer_8` in the variable namespace; it is not.

**Lines to change:** `convert_kwyk.py:11-13` (upgrade the provenance claim);
`scripts/kwyk_reproduction/ARCHITECTURE.md:115` (clarify that "Layer 8" is
positional, and its variables live under `logits/`, not `layer_8/`).

---

## 4. Numerical validation against the live TF graph

Static inventories cannot prove an axis permutation is right — only that shapes are
consistent. So the mapping was checked end-to-end against the actual TF graph.

**Determinism screening.** The SavedModel signature is
`volume (-1,32,32,32,1) → logits (-1,32,32,32,50)`. Running each model twice on an
identical seeded input:

| model | max&#124;run0 − run1&#124; | verdict |
|---|---|---|
| `all_50_wn` (MAP) | **0.000000** | deterministic — usable as reference |
| `all_50_bwn_09_multi` (BD) | 39.93 | stochastic |
| `all_50_bvwn_multi_prior` (SSD) | 11.52 | stochastic |

Only the MAP model admits a deterministic comparison; this is consistent with
ARCHITECTURE.md's "MC at inference: No" for `bwn`.

**Parity.** MAP weights converted with `--create-bias`, then compared against the
TF logits for the same input (`NDHWC`↔`NCDHW` permuted, PyTorch run with
`mc=False`):

```
TF   logits   mean -2.289535   std 3.125234
PT   logits   mean -2.289528   std 3.125223
max |diff|  = 9.918213e-05
mean|diff|  = 1.151648e-05
rel  error  = 3.693623e-06
argmax agreement = 100.0000%
```

A relative error of ~3.7e-06 over 7 dilated conv layers plus the collapsed output
layer is consistent with float32 accumulation and nothing else. **The axis
permutation, the weight-norm collapse, the 1-based↔0-based index mapping, the bias
copy, and the logits mapping are jointly correct.**

**One honest caveat about `argmax`.** The bias-dropped run in **D1** *also* scored
100% argmax agreement despite `max|Δ| = 19.77`. On this synthetic Gaussian input
one class dominates almost everywhere, so argmax is insensitive. **Argmax
agreement must not be used as the parity criterion** — the logit-level error is
the meaningful signal. A parity check on real T1 data would be a stronger test
still.

---

## 5. Summary

| Claim | Verdict |
|---|---|
| (a) conv `[k,k,k,in,out]`, `transpose(4,3,0,1,2)` | **CONFIRMED** (structural + numerical) |
| (b) `g` is `[1,1,1,1,out]` or `[out]` | **CONFIRMED** — always 5-D; 1-D branch unexercised |
| (c) index base, matches `_n_layers` | **CONFIRMED** — base 1, span 7 = `_n_layers` 7 |
| (d) `bias_m`/`bias_a` present every layer | **CONFIRMED** — 24/24 across 3 models |
| (e) `p` is a probability, not a logit | **CONFIRMED** — all 672 values in [0.61, 0.95] |

The mapping is **correct as implemented**, and verified numerically to ~4e-06
relative error. The defects are not in the transforms but in the **defaults and
the discarded data**: D1 (bias dropped by default, ~200,000× worse output, warning
only, and the built-in parity gate does not catch it) is the one that will bite a
user first. D2 is the one that matters most for the SSD model's stated purpose.

### Reproduction

```bash
docker pull neuronets/kwyk:latest-cpu
# variable inventory
docker run --rm --platform linux/amd64 --entrypoint python neuronets/kwyk:latest-cpu -c \
  "import tensorflow as tf; r=tf.train.NewCheckpointReader(
   '/opt/kwyk/saved_models/all_50_bvwn_multi_prior/1556816070/variables/variables');
   [print(k, r.get_variable_to_shape_map()[k]) for k in sorted(r.get_variable_to_shape_map())]"
```
Note the image is `linux/amd64` (emulated on arm64) and has an ENTRYPOINT — it must
be overridden with `--entrypoint python`.

---

## 6. Resolutions (implemented after the report)

All six discrepancies were fixed and re-validated. Where a discrepancy offered
two remedies, the choice and its rationale are recorded.

| # | Fix | Where |
|---|---|---|
| D1 | Biases are now imported **by default**; `--create-bias` replaced by an explicit `--drop-bias` opt-out whose warning quotes the measured max-~20 logit shift. `convert(create_bias=True)` is the new default. The false "checkpoints are bias-free" docstring was corrected. | `convert_kwyk.py` (`_build_arg_parser`, `convert`, `build_state_dict` warning), `kwyk_meshnet.py` (`bias` param docstring) |
| D2 | **Architectural fix chosen** (not the log-upgrade alternative): `KWYKMeshNet.classifier` is now an `FFGConv3d(kernel_size=1, bias=True)`, matching the TF `logits/` layer; `forward` passes `mc=mc_vwn` to it. The converter maps all five `logits/conv3d/*` variables 1:1 — including `bias_a`, which previously had no constant and was never read. Nothing is discarded. Chosen because the model is a *reproduction* of kwyk, the TF original **is** variational at the output, and no released checkpoint existed to break (all files pre-release). | `kwyk_meshnet.py` (classifier + forward), `convert_kwyk.py` (`_TF_LOGITS_BIAS_A`, `_add_classifier`, parity extended to the classifier) |
| D3 | Clamp widened to `[0.01, 0.99]` **in both places together**: `ConcreteDropout3d.p` and the converter, with module constants `CONCRETE_P_MIN/MAX` exposed in `vwn_layers.py` and a unit test (`test_clamp_constants_match_model`) enforcing the sync. `_p_to_logit` now logs a warning with the clipped count whenever any value falls outside the range. | `vwn_layers.py`, `convert_kwyk.py`, `test_convert_kwyk.py` |
| D4 | The 1-D `g` branch logs when taken (it never fires on real checkpoints). | `convert_kwyk.py` (`_g_tf_to_torch`) |
| D5 | `main()` writes a `<out stem>.provenance.json` sidecar recording the source path, its SHA-256 (the only way to distinguish the structurally identical MAP/BD checkpoints), the architecture args (not recoverable from a `state_dict`), and converter/library versions. | `convert_kwyk.py` (`_write_provenance`, `_sha256_of_file`) |
| D6 | Converter docstring now claims verification against the live container + live TF graph (not "ARCHITECTURE.md and source snippets"); `ARCHITECTURE.md` clarifies that "Layer 8" is positional and its variables live under `logits/conv3d/*`, not `layer_8/`. | `convert_kwyk.py` module docstring, `scripts/kwyk_reproduction/ARCHITECTURE.md` |

### Post-fix re-validation (real weights, live TF graph)

| check | result |
|---|---|
| SSD conversion, **no flags** (new defaults) | strict load ✓, parity ✓ over **8** conv layers *(incl. classifier — previously 7)* |
| D3: `max\|tf_p − model.p\|` over all 672 channels | **5.96e-08** (float32 round-trip only; was 0.004 one-signed clipping on 89% of channels) |
| D3: model `p` max | **0.954055** — exactly the TF value, no longer saturated at 0.95 |
| D2: converted SSD, deterministic path | bit-reproducible (max diff 0.0) |
| D2: converted SSD, MC path run-to-run | **max diff 10.6** — same order as the TF original's measured 11.5; the old plain-conv classifier contributed 0 |
| D1/TF-graph parity: converted MAP vs live TF logits | **max\|diff\| = 9.54e-05**, mean 1.15e-05 — unchanged from the pre-fix 9.92e-05 baseline |
| D5: provenance sidecar | written, carries source SHA-256 + arch args |
| Test suite | `test_convert_kwyk.py`: 17 passed (13 original + 4 new); full unit suite 377 passed with only the pre-existing unrelated `test_croissant` failure |

One deliberate consequence of D2 to be aware of: from-scratch `KWYKMeshNet`
training now treats the output layer variationally too (it contributes to
`kl_divergence()` and samples under `mc_vwn=True`). This is *more* faithful to
McClure et al. and to the TF implementation, but it does change the training
objective relative to the previous plain-conv classifier.

### Second multi-agent re-verification (post-fix)

The full three-agent workflow (PyTorch / TF / converter, independently
inventoried, then cross-checked with fresh numerical runs) was re-executed
after the fixes. Verdict: **D1–D6 all confirmed fixed**, with the TF ground
truth re-confirmed unchanged (41/41/48 variables, `logits/` namespace, p in
[0.607895, 0.954055]) and fresh conversions reproducing every §6 number
(MAP vs live TF 9.54e-05; p preserved to 5.96e-08; SSD MC output variance
present; sidecars distinguishing MAP/SSD by content hash; `--drop-bias`
warning firing and recording `bias: false`).

The re-run caught **two residual stale-doc fragments**, both fixed on the
spot:

1. The `kwyk_meshnet()` *factory* docstring still claimed "``bias`` defaults
   to ``False`` to match the published kwyk checkpoints" — the same false
   statement D1 removed from the class docstring. Corrected.
2. A test docstring still referenced the removed `--create-bias` flag.
   Corrected.

A repo sweep for further "bias-free"/"--create-bias" claims found only
accurate usages (descriptions of model configurations, not checkpoint
claims).
