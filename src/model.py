"""CAMoE modules corresponding to manuscript Eqs. (1)--(30)."""

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn

from .config import CAMoEConfig


MODALITIES = ("A", "V", "T")


def padding_mask(lengths: Tensor, max_length: int) -> Tensor:
    """Return a True-at-padding mask of shape [B, max_length]."""

    if lengths.ndim != 1:
        raise ValueError("lengths must have shape [batch]")
    if torch.any(lengths < 1) or torch.any(lengths > max_length):
        raise ValueError("each sequence length must be in [1, padded_length]")
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) >= lengths.unsqueeze(1)


def last_valid_state(sequence: Tensor, lengths: Tensor) -> Tensor:
    """Select H_m[length_m - 1, :] for every sample (correct zero-based index)."""

    batch = torch.arange(sequence.size(0), device=sequence.device)
    return sequence[batch, lengths - 1]


class ResidualProjection(nn.Module):
    """Two-layer residual projection used in Eqs. (4), (8), and after Eq. (20)."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.linear2(self.dropout(self.activation(self.linear1(x))))


class ModalityExpert(nn.Module):
    """Projection, temporal convolution, Transformer, and Eq. (3a) pooling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)  # Eq. (1)
        self.temporal_conv = nn.Conv1d(  # Eq. (2), T'_m = T_m - kernel + 1
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=0,
        )
        self.kernel_size = kernel_size
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, U_m: Tensor, lengths_m: Tensor) -> Dict[str, Tensor]:
        if U_m.ndim != 3:
            raise ValueError("U_m must have shape [batch, time, feature_dim]")
        if torch.any(lengths_m < self.kernel_size):
            raise ValueError("each valid sequence must be at least as long as the convolution kernel")
        X_m = self.input_projection(U_m)
        X_m = self.temporal_conv(X_m.transpose(1, 2)).transpose(1, 2)
        lengths_prime = lengths_m - self.kernel_size + 1
        mask_m = padding_mask(lengths_prime, X_m.size(1))
        H_m = self.transformer(X_m, src_key_padding_mask=mask_m)  # Eq. (3)
        r_m = last_valid_state(H_m, lengths_prime)  # Eq. (3a), zero-based T'_m-1
        return {
            "X": X_m,
            "H": H_m,
            "r": r_m,
            "mask": mask_m,
            "lengths_prime": lengths_prime,
        }


class UnimodalAuxiliaryRefinement(nn.Module):
    """UARM: Eq. (4) refined feature F_m and Eq. (5) prediction y_m."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.refine = ResidualProjection(dim, dropout)
        self.regressor = nn.Linear(dim, 1)

    def forward(self, r_m: Tensor) -> Dict[str, Tensor]:
        F_m = self.refine(r_m)
        y_m = self.regressor(F_m).squeeze(-1)
        return {"F": F_m, "y": y_m}


class AdaptiveTemperatureRouter(nn.Module):
    """ATR implementing Eqs. (8)--(14), including Eq. (10) LayerNorm."""

    def __init__(self, cfg: CAMoEConfig) -> None:
        super().__init__()
        d = cfg.hidden_dim
        self.routing_projections = nn.ModuleDict(
            {m: ResidualProjection(d, cfg.dropout_output) for m in MODALITIES}
        )
        self.linear1 = nn.Linear(3 * d, cfg.router_hidden_dim)
        self.layer_norm = nn.LayerNorm(cfg.router_hidden_dim)
        self.linear2 = nn.Linear(cfg.router_hidden_dim, 3)
        self.temperature_mlp = nn.Sequential(
            nn.Linear(3 * d, cfg.router_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.router_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.tau_min = cfg.tau_min
        self.tau_max = cfg.tau_max

    def forward(self, F: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        G = {m: self.routing_projections[m](F[m]) for m in MODALITIES}  # Eq. (8)
        M = torch.cat([G[m] for m in MODALITIES], dim=-1)  # Eq. (9): A,V,T
        h = torch.relu(self.layer_norm(self.linear1(M)))  # Eq. (10)
        ell = self.linear2(h)  # Eq. (11)
        tau_ratio = self.temperature_mlp(M)
        tau = self.tau_min + tau_ratio * (self.tau_max - self.tau_min)  # Eq. (12)
        w = torch.softmax(ell / tau, dim=-1)  # Eqs. (13)--(14)
        return {"G": G, "M": M, "h": h, "ell": ell, "tau": tau, "w": w}


class CrossModalAlignment(nn.Module):
    """Literal transcription of manuscript Eqs. (22)--(26).

    Eq. (4) defines F_m as fixed-dimensional, so it is represented as one
    key/value position here. This makes A_m have one position. The resulting
    degeneracy is retained deliberately instead of replacing F_m with H_m.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.W_Q = nn.ModuleDict({m: nn.Linear(dim, dim, bias=False) for m in MODALITIES})
        self.W_K = nn.ModuleDict({m: nn.Linear(dim, dim, bias=False) for m in MODALITIES})
        self.W_V = nn.ModuleDict({m: nn.Linear(dim, dim, bias=False) for m in MODALITIES})
        self.scale = dim ** 0.5

    def forward(
        self,
        c_fusion: Tensor,
        F: Mapping[str, Tensor],
    ) -> Dict[str, Dict[str, Tensor]]:
        z: Dict[str, Tensor] = {}
        A: Dict[str, Tensor] = {}
        for m in MODALITIES:
            Q_m = self.W_Q[m](c_fusion).unsqueeze(1)  # Eq. (22), [B,1,d]
            K_m = self.W_K[m](F[m]).unsqueeze(1)  # Eq. (23), [B,1,d]
            V_m = self.W_V[m](F[m]).unsqueeze(1)  # Eq. (23), [B,1,d]
            scores = torch.matmul(Q_m, K_m.transpose(-2, -1)) / self.scale
            A[m] = torch.softmax(scores, dim=-1).squeeze(1)  # Eqs. (24)--(25)
            z[m] = torch.matmul(A[m].unsqueeze(1), V_m).squeeze(1)  # Eq. (26)
        return {"z": z, "A": A}


class CAMoE(nn.Module):
    """Complete CAMoE forward graph.

    Inputs are padded U_T, U_A and U_V tensors plus their valid lengths. The
    output dictionary deliberately exposes the manuscript variables so that
    every loss and claimed mechanism can be inspected independently.
    """

    def __init__(self, config: Optional[CAMoEConfig] = None) -> None:
        super().__init__()
        self.config = config or CAMoEConfig()
        cfg = self.config
        expert_args = {
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
        }
        self.experts = nn.ModuleDict(
            {
                "T": ModalityExpert(
                    cfg.d_text,
                    kernel_size=cfg.conv_kernel_text,
                    dropout=cfg.dropout_text,
                    **expert_args,
                ),
                "A": ModalityExpert(
                    cfg.d_audio,
                    kernel_size=cfg.conv_kernel_audio,
                    dropout=cfg.dropout_audio,
                    **expert_args,
                ),
                "V": ModalityExpert(
                    cfg.d_visual,
                    kernel_size=cfg.conv_kernel_visual,
                    dropout=cfg.dropout_visual,
                    **expert_args,
                ),
            }
        )
        dropouts = {
            "T": cfg.dropout_text,
            "A": cfg.dropout_audio,
            "V": cfg.dropout_visual,
        }
        self.uarm = nn.ModuleDict(
            {m: UnimodalAuxiliaryRefinement(cfg.hidden_dim, dropouts[m]) for m in MODALITIES}
        )
        self.atr = AdaptiveTemperatureRouter(cfg)
        self.fusion_refinement = ResidualProjection(cfg.hidden_dim, cfg.dropout_output)
        self.fusion_regressor = nn.Linear(cfg.hidden_dim, 1)
        self.cam = CrossModalAlignment(cfg.hidden_dim)

    def forward(
        self,
        U_T: Tensor,
        U_A: Tensor,
        U_V: Tensor,
        lengths_T: Tensor,
        lengths_A: Tensor,
        lengths_V: Tensor,
    ) -> Dict[str, object]:
        U = {"T": U_T, "A": U_A, "V": U_V}
        lengths = {"T": lengths_T, "A": lengths_A, "V": lengths_V}

        encoded = {m: self.experts[m](U[m], lengths[m]) for m in MODALITIES}
        H = {m: encoded[m]["H"] for m in MODALITIES}
        r = {m: encoded[m]["r"] for m in MODALITIES}
        masks = {m: encoded[m]["mask"] for m in MODALITIES}

        refined = {m: self.uarm[m](r[m]) for m in MODALITIES}
        F = {m: refined[m]["F"] for m in MODALITIES}
        y_unimodal = {m: refined[m]["y"] for m in MODALITIES}

        routing = self.atr(F)
        w = routing["w"]  # columns follow [A, V, T]
        c_fusion = sum(w[:, i : i + 1] * F[m] for i, m in enumerate(MODALITIES))
        c_refined = self.fusion_refinement(c_fusion)
        y_hat = self.fusion_regressor(c_refined).squeeze(-1)
        alignment = self.cam(c_fusion, F)

        return {
            "H": H,
            "r": r,
            "F": F,
            "G": routing["G"],
            "M": routing["M"],
            "h": routing["h"],
            "ell": routing["ell"],
            "tau": routing["tau"],
            "w": w,
            "c_fusion": c_fusion,
            "c_refined": c_refined,
            "y_unimodal": y_unimodal,
            "y_hat": y_hat,
            "z": alignment["z"],
            "A": alignment["A"],
            "masks": masks,
        }
