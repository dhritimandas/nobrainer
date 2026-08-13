"""Unit tests for nobrainer.datasets.convert_kwyk.

No TensorFlow or network dependency: a stub model matching the KWYKMeshNet
interface is used as ground truth. Fabricated TF-layout variables are derived
from it, converted back, and checked for round-trip equality.

If these tests fail after wiring the real model, the likely causes are:
- ConcreteDropout logit attribute is not `dropout.p_logit` (update the template).
- The kwyk factory cannot build bias-capable convs (see test_create_bias_*).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nobrainer.datasets import convert_kwyk as ck

# --- Stub model mirroring the real KWYKMeshNet interface -----------------------


class _StubFFGConv3d(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, k: int = 3, dilation: int = 1, bias: bool = False
    ) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_ch, out_ch
        self.kernel_size, self.stride = k, 1
        self.padding, self.dilation = dilation, dilation
        ws = (out_ch, in_ch, k, k, k)
        self.v = nn.Parameter(torch.empty(ws))
        nn.init.kaiming_normal_(self.v)
        self.g = nn.Parameter(torch.full((out_ch, 1, 1, 1, 1), math.sqrt(2.0)))
        self.kernel_a = nn.Parameter(torch.full(ws, 1e-4))
        if bias:
            self.bias_m = nn.Parameter(torch.zeros(out_ch))
            self.bias_a = nn.Parameter(torch.full((out_ch,), 1e-4))
        else:
            self.register_parameter("bias_m", None)
            self.register_parameter("bias_a", None)

    @property
    def kernel_m(self) -> torch.Tensor:
        v_norm = F.normalize(self.v.flatten(1), dim=1).view_as(self.v)
        return self.g * v_norm


class _StubDropout(nn.Module):
    def __init__(self, out_ch: int) -> None:
        super().__init__()
        self.p_logit = nn.Parameter(torch.zeros(out_ch))


class _StubLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, bias: bool) -> None:
        super().__init__()
        self.conv = _StubFFGConv3d(in_ch, out_ch, 3, dilation, bias)
        self.dropout = _StubDropout(out_ch)


class _StubKWYK(nn.Module):
    def __init__(self, bias: bool = False) -> None:
        super().__init__()
        dilations = [1, 2, 4]
        self._n_layers = len(dilations)
        for i, d in enumerate(dilations):
            in_ch = 1 if i == 0 else 8
            setattr(self, f"layer_{i}", _StubLayer(in_ch, 8, d, bias))
        # Mirror the real KWYKMeshNet: the classifier is itself an FFG conv
        # (v/g/kernel_a/bias_m/bias_a), matching the TF logits/ layer 1:1.
        # Without this the stub diverges from the real model in exactly the
        # spot the converter must fill, and the suite would stay green while
        # main() fails on the real architecture.
        self.classifier = _StubFFGConv3d(8, 2, k=1, dilation=1, bias=True)


# --- Fixtures ------------------------------------------------------------------


@pytest.fixture
def ground_truth() -> _StubKWYK:
    torch.manual_seed(1)
    return _StubKWYK(bias=True)


@pytest.fixture
def tf_vars(ground_truth: _StubKWYK) -> dict[str, np.ndarray]:
    """Fabricate 1-based TF-layout variables from the ground-truth model."""
    out: dict[str, np.ndarray] = {}
    for i in range(ground_truth._n_layers):
        conv = getattr(ground_truth, f"layer_{i}").conv
        # torch [out,in,k,k,k] -> TF [k,k,k,in,out] (inverse of _CONV_PERM)
        out[f"layer_{i + 1}/conv3d/v"] = np.transpose(
            conv.v.detach().numpy(), (2, 3, 4, 1, 0)
        )
        out[f"layer_{i + 1}/conv3d/g"] = np.transpose(
            conv.g.detach().numpy(), (2, 3, 4, 1, 0)
        )
        out[f"layer_{i + 1}/conv3d/kernel_a"] = np.transpose(
            conv.kernel_a.detach().numpy(), (2, 3, 4, 1, 0)
        )
        out[f"layer_{i + 1}/conv3d/bias_m"] = conv.bias_m.detach().numpy()
        out[f"layer_{i + 1}/conv3d/bias_a"] = conv.bias_a.detach().numpy()
        p = torch.sigmoid(getattr(ground_truth, f"layer_{i}").dropout.p_logit)
        out[f"layer_{i + 1}/concrete_dropout/p"] = p.detach().numpy()

    # Output layer: its own ``logits/`` namespace, itself a full VWN conv
    # with NO concrete_dropout sibling (verified against
    # neuronets/kwyk:latest-cpu). All five variables map 1:1 onto the FFG
    # classifier, so fabricate them from the stub classifier's own params.
    clf = ground_truth.classifier
    out["logits/conv3d/v"] = np.transpose(clf.v.detach().numpy(), (2, 3, 4, 1, 0))
    out["logits/conv3d/g"] = np.transpose(clf.g.detach().numpy(), (2, 3, 4, 1, 0))
    out["logits/conv3d/kernel_a"] = np.transpose(
        clf.kernel_a.detach().numpy(), (2, 3, 4, 1, 0)
    )
    out["logits/conv3d/bias_m"] = clf.bias_m.detach().numpy()
    out["logits/conv3d/bias_a"] = clf.bias_a.detach().numpy()
    return out


@pytest.fixture
def npz_path(tf_vars: dict[str, np.ndarray], tmp_path: Path) -> Path:
    path = tmp_path / "fake_tf.npz"
    np.savez(path, **tf_vars)
    return path


# --- Tests: happy path ---------------------------------------------------------


def test_convert_roundtrip_matches_ground_truth(ground_truth, npz_path):
    """Converted model reproduces the ground-truth mean path within tolerance.

    Deliberately relies on convert()'s DEFAULT create_bias -- which must keep
    biases, since every published checkpoint has them (report D1).
    """
    dst = _StubKWYK(bias=True)
    ck.convert(dst, npz=npz_path, run_parity=True)
    x = torch.randn(1, 1, 8, 8, 8)
    with torch.no_grad():
        a = ground_truth.layer_0.conv
        b = dst.layer_0.conv
        out_a = F.conv3d(x, a.kernel_m, a.bias_m, a.stride, a.padding, a.dilation)
        out_b = F.conv3d(x, b.kernel_m, b.bias_m, b.stride, b.padding, b.dilation)
    torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_p_logit_recovered(ground_truth, npz_path):
    dst = _StubKWYK(bias=True)
    ck.convert(dst, npz=npz_path, create_bias=True, run_parity=False)
    torch.testing.assert_close(
        dst.layer_0.dropout.p_logit,
        ground_truth.layer_0.dropout.p_logit,
        atol=1e-5,
        rtol=1e-5,
    )


def test_classifier_recovered(ground_truth, npz_path):
    """The VWN logits layer maps 1:1 onto the FFG classifier -- all five params."""
    dst = _StubKWYK(bias=True)
    ck.convert(dst, npz=npz_path, create_bias=True, run_parity=False)
    for attr in ("v", "g", "kernel_a", "bias_m", "bias_a"):
        torch.testing.assert_close(
            getattr(dst.classifier, attr),
            getattr(ground_truth.classifier, attr),
            atol=1e-6,
            rtol=1e-6,
        )


def test_missing_logits_layer_rejected(tf_vars, tmp_path):
    """A checkpoint without the logits layer must fail loudly, not partially load."""
    hidden_only = {k: v for k, v in tf_vars.items() if not k.startswith("logits/")}
    path = tmp_path / "no_logits.npz"
    np.savez(path, **hidden_only)
    dst = _StubKWYK(bias=True)
    with pytest.raises(ck.ConversionError, match="no logits layer"):
        ck.convert(dst, npz=path, create_bias=True, run_parity=False)


def test_detect_index_base_one(tf_vars):
    assert ck.detect_layer_index_base(tf_vars, 3) == 1


def test_g_1d_handled():
    g = np.random.randn(8).astype(np.float32)
    out = ck._g_tf_to_torch(g, 8)
    assert out.shape == (8, 1, 1, 1, 1)


def test_g_5d_handled():
    g = np.random.randn(1, 1, 1, 1, 8).astype(np.float32)
    out = ck._g_tf_to_torch(g, 8)
    assert out.shape == (8, 1, 1, 1, 1)


# --- Tests: guard rails (must fail loudly) -------------------------------------


def test_create_bias_on_biasless_model_rejected(npz_path):
    """Model built bias=False + bias import (the default) must fail strict load.

    The mismatch guard: importing biases into a bias-free model produces
    unexpected keys, which strict=True rejects loudly.
    """
    dst = _StubKWYK(bias=False)
    with pytest.raises(RuntimeError):  # strict load raises on unexpected keys
        ck.convert(dst, npz=npz_path, create_bias=True, run_parity=False)


def test_layer_count_mismatch_caught(tf_vars):
    with pytest.raises(ck.ConversionError):
        ck.detect_layer_index_base(tf_vars, 5)  # npz has 3 layers


def test_conv_wrong_ndim_rejected():
    with pytest.raises(ck.ConversionError):
        ck._conv_tf_to_torch(np.zeros((3, 3, 3)))  # 3-D, not 5-D


def test_parity_detects_transform_error(ground_truth, npz_path, monkeypatch):
    """If the conv transpose is wrong, parity must catch it.

    The offline check validates the mapping transform (axis order, weight-norm),
    not source-data integrity: corrupting a TF var identically on both the
    recompute and load paths would move together and pass. So we inject a wrong
    permutation and confirm parity fails loudly.
    """
    dst = _StubKWYK(bias=True)
    # Patch the conv permutation to an incorrect axis order.
    monkeypatch.setattr(ck, "_CONV_PERM", (0, 1, 2, 3, 4))
    with pytest.raises(ck.ConversionError):
        ck.convert(dst, npz=npz_path, create_bias=True, run_parity=True)


def test_both_sources_rejected(npz_path):
    dst = _StubKWYK(bias=True)
    with pytest.raises(ck.ConversionError, match="exactly one"):
        ck.convert(dst, npz=npz_path, tf_path=npz_path, create_bias=True)


def test_no_source_rejected():
    dst = _StubKWYK(bias=True)
    with pytest.raises(ck.ConversionError, match="exactly one"):
        ck.convert(dst)


# --- Tests: discrepancy fixes (docs/kwyk_mapping_verification.md) --------------


def test_clamp_constants_match_model():
    """The converter's p clamp MUST equal ConcreteDropout3d's (report D3).

    The two are literal copies (the converter avoids importing nobrainer at
    module scope); this test is the sync mechanism -- widening one without
    the other silently reintroduces the clipping bug.
    """
    from nobrainer.models.bayesian.vwn_layers import CONCRETE_P_MAX, CONCRETE_P_MIN

    assert ck._CONCRETE_P_MIN == CONCRETE_P_MIN
    assert ck._CONCRETE_P_MAX == CONCRETE_P_MAX


def test_published_p_range_survives_conversion():
    """p values up to 0.954 (the real SSD max) must NOT be clipped (report D3)."""
    p = np.array([0.607895, 0.9, 0.953, 0.954055], dtype=np.float32)
    logit = ck._p_to_logit(p)
    recovered = 1.0 / (1.0 + np.exp(-logit))
    np.testing.assert_allclose(recovered, p, rtol=1e-5, atol=1e-6)


def test_main_end_to_end_writes_weights_and_provenance(tmp_path):
    """main() converts a real (tiny) KWYKMeshNet checkpoint and records
    provenance -- the sidecar is the only durable record of which source
    model a .pth came from, since MAP and BD checkpoints are structurally
    indistinguishable (report D5). Also exercises report D1's new default:
    no bias flag passed, biases kept."""
    import json

    from nobrainer.models import get as get_model

    torch.manual_seed(3)
    src = get_model("kwyk_meshnet")(
        n_classes=2, filters=8, receptive_field=37, dropout_type="concrete", bias=True
    )
    tf_vars: dict = {}
    for i in range(src._n_layers):
        conv = getattr(src, f"layer_{i}").conv
        tf_vars[f"layer_{i + 1}/conv3d/v"] = np.transpose(
            conv.v.detach().numpy(), (2, 3, 4, 1, 0)
        )
        tf_vars[f"layer_{i + 1}/conv3d/g"] = np.transpose(
            conv.g.detach().numpy(), (2, 3, 4, 1, 0)
        )
        tf_vars[f"layer_{i + 1}/conv3d/kernel_a"] = np.transpose(
            conv.kernel_a.detach().numpy(), (2, 3, 4, 1, 0)
        )
        tf_vars[f"layer_{i + 1}/conv3d/bias_m"] = conv.bias_m.detach().numpy()
        tf_vars[f"layer_{i + 1}/conv3d/bias_a"] = conv.bias_a.detach().numpy()
        p = torch.sigmoid(getattr(src, f"layer_{i}").dropout.p_logit)
        tf_vars[f"layer_{i + 1}/concrete_dropout/p"] = p.detach().numpy()
    clf = src.classifier
    tf_vars["logits/conv3d/v"] = np.transpose(clf.v.detach().numpy(), (2, 3, 4, 1, 0))
    tf_vars["logits/conv3d/g"] = np.transpose(clf.g.detach().numpy(), (2, 3, 4, 1, 0))
    tf_vars["logits/conv3d/kernel_a"] = np.transpose(
        clf.kernel_a.detach().numpy(), (2, 3, 4, 1, 0)
    )
    tf_vars["logits/conv3d/bias_m"] = clf.bias_m.detach().numpy()
    tf_vars["logits/conv3d/bias_a"] = clf.bias_a.detach().numpy()

    npz = tmp_path / "tiny_kwyk.npz"
    np.savez(npz, **tf_vars)
    out = tmp_path / "converted.pth"

    rc = ck.main(
        [
            "--npz",
            str(npz),
            "--n-classes",
            "2",
            "--filters",
            "8",
            "--receptive-field",
            "37",
            "--dropout-type",
            "concrete",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()

    prov_path = tmp_path / "converted.provenance.json"
    assert prov_path.exists()
    prov = json.loads(prov_path.read_text())
    assert prov["source"] == str(npz)
    assert prov["source_sha256"] == ck._sha256_of_file(npz)
    assert prov["bias"] is True  # D1: kept by default, no flag passed
    assert prov["dropout_type"] == "concrete"

    # The written weights round-trip into a fresh model and reproduce the
    # source's deterministic forward exactly (mean path).
    dst = get_model("kwyk_meshnet")(
        n_classes=2, filters=8, receptive_field=37, dropout_type="concrete", bias=True
    )
    dst.load_state_dict(torch.load(out, weights_only=True), strict=True)
    src.eval()
    dst.eval()
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(
            src(x, mc=False), dst(x, mc=False), atol=1e-6, rtol=1e-6
        )


def test_real_classifier_contributes_mc_variance():
    """The FFG classifier must sample under mc=True and be exactly
    deterministic under mc=False -- the point of report D2's fix (the old
    plain-conv classifier contributed zero output variance)."""
    from nobrainer.models.bayesian.vwn_layers import FFGConv3d

    torch.manual_seed(5)
    clf = FFGConv3d(8, 2, kernel_size=1, bias=True, sigma_init=0.1)
    clf.eval()
    x = torch.randn(1, 8, 4, 4, 4)
    with torch.no_grad():
        a = clf(x, mc=True)
        b = clf(x, mc=True)
        det1 = clf(x, mc=False)
        det2 = clf(x, mc=False)
    assert not torch.allclose(a, b), "mc=True must sample"
    torch.testing.assert_close(det1, det2, atol=0.0, rtol=0.0)
