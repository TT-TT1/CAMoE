"""Adaptive Temperature Router (ATR), manuscript Eqs. (8)--(14).

The modality order is always [A, V, T] (acoustic, visual, textual). Eq. (10)
is implemented with ``LayerNorm`` rather than L2 vector normalization.
"""

from typing import Dict

import torch
from torch import Tensor, nn


MODALITY_ORDER = ("A", "V", "T")


class RoutingProjection(nn.Module):
    """Modality-specific residual projection in Eq. (8).

    G_m = Proj2_m(Dropout(ReLU(Proj1_m(F_m)))) + F_m
    """

    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj1 = nn.Linear(feature_dim, feature_dim)
        self.proj2 = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, F_m: Tensor) -> Tensor:
        projected = self.proj1(F_m)
        projected = self.relu(projected)
        projected = self.dropout(projected)
        return self.proj2(projected) + F_m


class AdaptiveTemperatureRouter(nn.Module):
    """Reliability-aware router defined in manuscript Section 3.4.

    Parameters
    ----------
    feature_dim:
        Common hidden dimension d of every refined expert representation F_m.
    router_hidden_dim:
        Hidden dimension of the two-layer routing MLP.
    dropout:
        Dropout probability in each Eq. (8) routing projection.
    tau_min, tau_max:
        Lower and upper bounds of the sample-specific temperature in Eq. (12).

    Inputs
    ------
    F_A, F_V, F_T:
        UARM-refined acoustic, visual, and textual expert representations,
        each with shape [B, d].

    Returns
    -------
    A dictionary exposing the manuscript variables G_A/G_V/G_T, M, h, ell,
    tau, and w.  ``ell`` and ``w`` use column order [A, V, T].
    """

    def __init__(
        self,
        feature_dim: int,
        router_hidden_dim: int,
        dropout: float = 0.5,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or router_hidden_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < tau_min <= tau_max:
            raise ValueError("temperature bounds must satisfy 0 < tau_min <= tau_max")

        self.routing_projection_A = RoutingProjection(feature_dim, dropout)
        self.routing_projection_V = RoutingProjection(feature_dim, dropout)
        self.routing_projection_T = RoutingProjection(feature_dim, dropout)

        # Eqs. (10)--(11): h = ReLU(LayerNorm(Linear1(M))); ell = Linear2(h).
        self.linear1 = nn.Linear(3 * feature_dim, router_hidden_dim)
        self.layer_norm = nn.LayerNorm(router_hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(router_hidden_dim, 3)

        # Eq. (12): tau = sigmoid(MLP_tau(M)) * (tau_max-tau_min) + tau_min.
        self.temperature_mlp = nn.Sequential(
            nn.Linear(3 * feature_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

    def forward(self, F_A: Tensor, F_V: Tensor, F_T: Tensor) -> Dict[str, Tensor]:
        self._validate_inputs(F_A, F_V, F_T)

        # Eq. (8): routing-aware modality representations.
        G_A = self.routing_projection_A(F_A)
        G_V = self.routing_projection_V(F_V)
        G_T = self.routing_projection_T(F_T)

        # Eq. (9): fixed order [acoustic, visual, textual].
        M = torch.cat((G_A, G_V, G_T), dim=-1)

        # Eq. (10): normalization is performed by the LayerNorm declared above.
        h = self.relu(self.layer_norm(self.linear1(M)))

        # Eq. (11): three routing logits ell_A, ell_V, ell_T.
        ell = self.linear2(h)

        # Eq. (12): one bounded temperature for each utterance.
        tau_ratio = self.temperature_mlp(M)
        tau = self.tau_min + tau_ratio * (self.tau_max - self.tau_min)

        # Eqs. (13)--(14): temperature-scaled routing distribution.
        w = torch.softmax(ell / tau, dim=-1)

        return {
            "G_A": G_A,
            "G_V": G_V,
            "G_T": G_T,
            "M": M,
            "h": h,
            "ell": ell,
            "tau": tau,
            "w": w,
        }

    @staticmethod
    def _validate_inputs(F_A: Tensor, F_V: Tensor, F_T: Tensor) -> None:
        features = (F_A, F_V, F_T)
        if any(feature.ndim != 2 for feature in features):
            raise ValueError("F_A, F_V, and F_T must each have shape [B, d]")
        if not (F_A.shape == F_V.shape == F_T.shape):
            raise ValueError("F_A, F_V, and F_T must have identical shapes")


def compute_atr_losses(
    w: Tensor,
    y_A: Tensor,
    y_V: Tensor,
    y_T: Tensor,
    y: Tensor,
    gamma: float = 0.1,
    epsilon: float = 1e-8,
) -> Dict[str, Tensor]:
    """Compute the ATR regularization terms in Eqs. (15)--(19).

    ``w`` must follow [A, V, T] order. The KL direction is exactly
    ``KL(alpha || w)`` as specified by Eq. (18). The manuscript specifies no
    stop-gradient for ``alpha``, so this literal implementation does not add
    one.
    """

    y = y.reshape(-1)
    predictions = (y_A.reshape(-1), y_V.reshape(-1), y_T.reshape(-1))
    if w.ndim != 2 or w.shape[1] != 3:
        raise ValueError("w must have shape [B, 3] in [A, V, T] order")
    if any(prediction.shape != y.shape for prediction in predictions):
        raise ValueError("unimodal predictions and y must have shape [B]")
    if w.shape[0] != y.shape[0]:
        raise ValueError("w and predictions must have the same batch size")

    # Eq. (16): per-sample unimodal absolute errors [epsilon_A,epsilon_V,epsilon_T].
    epsilon_uni = torch.stack(
        [(prediction - y).abs() for prediction in predictions], dim=-1
    )

    # Eq. (15): negative entropy; minimizing it maximizes routing entropy.
    L_ety = (w * torch.log(w + epsilon)).sum(dim=-1).mean()

    # Eqs. (17)--(18): reliability target and KL(alpha || w).
    alpha = torch.softmax(-epsilon_uni, dim=-1)
    L_route = (
        alpha * torch.log((alpha + epsilon) / (w + epsilon))
    ).sum(dim=-1).mean()

    # Eq. (19).
    L_atr = L_ety + gamma * L_route

    return {
        "epsilon_uni": epsilon_uni,
        "alpha": alpha,
        "L_ety": L_ety,
        "L_route": L_route,
        "L_atr": L_atr,
    }


# Backward-compatible name for code that imports ``router`` from router.py.
# The implementation remains the Eq. (8)--(14) ATR above.
router = AdaptiveTemperatureRouter


__all__ = [
    "AdaptiveTemperatureRouter",
    "MODALITY_ORDER",
    "RoutingProjection",
    "compute_atr_losses",
    "router",
]
