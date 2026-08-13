"""KWYK MeshNet variants — matching McClure et al. (2019) architecture.

All three kwyk models use Fully Factorized Gaussian (FFG) convolutions
with learned per-weight μ and σ, and the local reparameterization trick
(Kingma et al. 2015).  They differ in the dropout layer:

* **bwn** / **bwn_multi**: FFG conv + Bernoulli dropout
  (``bwn`` disables dropout at inference; ``bwn_multi`` keeps it on)
* **bvwn_multi_prior**: FFG conv + Concrete dropout (learned per-filter rate)
  This is the "spike-and-slab dropout" (SSD) model from the paper.

Reference
---------
McClure P. et al., "Knowing What You Know in Brain Segmentation Using
Bayesian Deep Neural Networks", Front. Neuroinform. 2019.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nobrainer.models._constants import (  # noqa: E501
    DILATION_SCHEDULES as _DILATION_SCHEDULES,
)

from .vwn_layers import ConcreteDropout3d, FFGConv3d


class _VWNLayerBernoulli(nn.Module):
    """VWN conv + ReLU + Bernoulli dropout (bwn / bwn_multi)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilation: int,
        dropout_rate: float,
        sigma_init: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = FFGConv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=bias,
            sigma_init=sigma_init,
        )
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        mc_vwn: bool = True,
        mc_dropout: bool = True,
    ) -> torch.Tensor:
        # Original TF order: conv -> dropout -> relu (meshnetbwn.py:59-61)
        h = self.conv(x, mc=mc_vwn)
        if mc_dropout:
            h = self.dropout(h)
        return F.relu(h)


class _VWNLayerConcrete(nn.Module):
    """VWN conv + ReLU + Concrete dropout (bvwn_multi_prior)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilation: int,
        sigma_init: float,
        concrete_temperature: float = 0.02,
        concrete_init_p: float = 0.9,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = FFGConv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=bias,
            sigma_init=sigma_init,
        )
        self.dropout = ConcreteDropout3d(
            out_ch,
            temperature=concrete_temperature,
            init_p=concrete_init_p,
        )

    def forward(
        self,
        x: torch.Tensor,
        mc_vwn: bool = True,
        mc_dropout: bool = True,
    ) -> torch.Tensor:
        # Original TF order: conv -> dropout -> relu (meshnetbvwn.py:54-55)
        h = self.conv(x, mc=mc_vwn)
        h = self.dropout(h, mc=mc_dropout)
        return F.relu(h)


class KWYKMeshNet(nn.Module):
    """KWYK MeshNet with variational weight normalization.

    This is the architecture used in McClure et al. (2019).  All layers
    use VWN convolutions; the ``dropout_type`` parameter selects between
    Bernoulli (``"bernoulli"``) and Concrete (``"concrete"``) dropout.

    Parameters
    ----------
    n_classes : int
        Number of output segmentation classes.
    in_channels : int
        Number of input image channels.
    filters : int
        Feature-map count in all hidden layers.
    receptive_field : int
        One of ``37``, ``67``, ``129`` — selects the dilation schedule.
    dropout_type : str
        ``"bernoulli"`` for bwn/bwn_multi, ``"concrete"`` for bvwn_multi_prior.
    dropout_rate : float
        For Bernoulli dropout (ignored for concrete).
    sigma_init : float
        Initial value for weight sigma (default 1e-4, matching kwyk).
    concrete_temperature : float
        Temperature for concrete dropout (default 0.02).
    concrete_init_p : float
        Initial dropout probability for concrete dropout (default 0.9).
    bias : bool
        Whether the hidden-layer FFG convolutions carry a bias term
        (``bias_m`` / ``bias_a``).  Default ``False`` (the from-scratch
        training default).  Note the **published kwyk checkpoints DO carry
        ``bias_m``/``bias_a`` for every conv layer** — verified against the
        ``neuronets/kwyk`` container, see
        ``docs/kwyk_mapping_verification.md`` — so importing them requires
        ``bias=True`` (``nobrainer/datasets/convert_kwyk.py`` builds the
        model that way by default); a strict state_dict load rejects the
        bias keys against a bias-free model.  The output ``classifier`` is
        unaffected by this flag: it always carries its bias, matching both
        the TF ``logits/`` layer and the previous ``nn.Conv3d`` behaviour.
    """

    def __init__(
        self,
        n_classes: int = 1,
        in_channels: int = 1,
        filters: int = 71,
        receptive_field: int = 67,
        dropout_type: str = "bernoulli",
        dropout_rate: float = 0.25,
        sigma_init: float = 1e-4,
        concrete_temperature: float = 0.02,
        concrete_init_p: float = 0.9,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if receptive_field not in _DILATION_SCHEDULES:
            raise ValueError(
                f"receptive_field must be one of {list(_DILATION_SCHEDULES)}, "
                f"got {receptive_field}"
            )
        self.dropout_type = dropout_type
        dilations = _DILATION_SCHEDULES[receptive_field]
        self._n_layers = len(dilations)

        for i, dil in enumerate(dilations):
            in_ch = in_channels if i == 0 else filters
            if dropout_type == "concrete":
                layer = _VWNLayerConcrete(
                    in_ch,
                    filters,
                    dil,
                    sigma_init,
                    concrete_temperature,
                    concrete_init_p,
                    bias=bias,
                )
            else:
                layer = _VWNLayerBernoulli(
                    in_ch,
                    filters,
                    dil,
                    dropout_rate,
                    sigma_init,
                    bias=bias,
                )
            setattr(self, f"layer_{i}", layer)

        # The output layer is itself a VWN (FFG) conv, exactly like the TF
        # original's ``logits/conv3d/*`` layer (a full VWN conv with bias --
        # verified against the neuronets/kwyk container, see
        # docs/kwyk_mapping_verification.md, discrepancy D2). A plain
        # nn.Conv3d here could represent only the mean path, making MC
        # uncertainty systematically under-dispersed at the output. bias is
        # unconditional: the TF logits layer always has one, and so did the
        # previous nn.Conv3d default.
        self.classifier = FFGConv3d(
            filters, n_classes, kernel_size=1, bias=True, sigma_init=sigma_init
        )

    def forward(
        self,
        x: torch.Tensor,
        mc: bool | None = None,
        mc_vwn: bool = True,
        mc_dropout: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input ``(B, 1, D, H, W)``.
        mc : bool or None
            Legacy convenience flag.  If provided, sets both ``mc_vwn``
            and ``mc_dropout`` to the same value (backward compat).
        mc_vwn : bool
            If True, use stochastic VWN reparameterization.
            If False, use deterministic mean weights only.
        mc_dropout : bool
            If True, apply stochastic dropout.
            If False, skip dropout (Bernoulli) or use expectation (Concrete).

        Note
        ----
        The original TF bwn model trains with ``mc_vwn=False, mc_dropout=True``
        (deterministic weights + stochastic dropout).
        """
        if mc is not None:
            mc_vwn = mc
            mc_dropout = mc

        h = x
        for i in range(self._n_layers):
            h = getattr(self, f"layer_{i}")(h, mc_vwn=mc_vwn, mc_dropout=mc_dropout)
        # The classifier follows the same VWN sampling switch as the hidden
        # convs: stochastic under mc_vwn=True, mean path under mc_vwn=False.
        return self.classifier(h, mc=mc_vwn)

    def kl_divergence(self) -> torch.Tensor:
        """Sum KL divergence from all VWN conv layers."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, FFGConv3d):
                kl = kl + m.kl
        return kl

    def concrete_regularization(self) -> torch.Tensor:
        """Sum concrete dropout regularization (0 for bernoulli models)."""
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, ConcreteDropout3d):
                reg = reg + m.regularization()
        return reg


def kwyk_meshnet(
    n_classes: int = 1,
    in_channels: int = 1,
    filters: int = 71,
    receptive_field: int = 67,
    dropout_type: str = "bernoulli",
    dropout_rate: float = 0.25,
    sigma_init: float = 1e-4,
    concrete_temperature: float = 0.02,
    concrete_init_p: float = 0.9,
    bias: bool = False,
    **kwargs,
) -> KWYKMeshNet:
    """Factory function for :class:`KWYKMeshNet`.

    ``bias`` defaults to ``False`` (the from-scratch training default). The
    published kwyk checkpoints DO carry ``bias_m``/``bias_a`` for every conv
    layer (verified against the ``neuronets/kwyk`` container -- see
    ``docs/kwyk_mapping_verification.md``), so pass ``bias=True`` when
    importing them; ``nobrainer/datasets/convert_kwyk.py`` does so by default.
    """
    return KWYKMeshNet(
        n_classes=n_classes,
        in_channels=in_channels,
        filters=filters,
        receptive_field=receptive_field,
        dropout_type=dropout_type,
        dropout_rate=dropout_rate,
        sigma_init=sigma_init,
        concrete_temperature=concrete_temperature,
        concrete_init_p=concrete_init_p,
        bias=bias,
    )


__all__ = ["KWYKMeshNet", "kwyk_meshnet"]
