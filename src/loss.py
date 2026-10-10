"""Loss terms corresponding to manuscript Eqs. (6)--(7), (15)--(31)."""

from typing import Dict, Mapping

import torch
from torch import Tensor, nn

from .config import CAMoEConfig
from .model import MODALITIES


class CAMoELoss(nn.Module):
    """Compute the complete joint objective and expose every named component."""

    def __init__(self, config: CAMoEConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, output: Mapping[str, object], y: Tensor) -> Dict[str, Tensor]:
        cfg = self.config
        y = y.view(-1)
        y_hat = output["y_hat"]
        y_unimodal = output["y_unimodal"]
        w = output["w"]


        epsilon_uni = torch.stack(
            [(y_unimodal[m] - y).abs() for m in MODALITIES], dim=-1
        )
        L_uni = epsilon_uni.mean()
        L_multi = (y_hat - y).abs().mean()  # Eq. (21)

        L_ety = (w * torch.log(w.clamp_min(cfg.eps))).sum(dim=-1).mean()  # Eq. (15)
        alpha = torch.softmax(-epsilon_uni, dim=-1)  # Eq. (17)
   
        L_route = (
            alpha
            * (
                torch.log(alpha.clamp_min(cfg.eps))
                - torch.log(w.clamp_min(cfg.eps))
            )
        ).sum(dim=-1).mean()
        L_atr = L_ety + cfg.gamma * L_route  # Eq. (19)

        beta = torch.softmax(-epsilon_uni, dim=-1)  # Eq. (27)
        c_anchor = output["c_fusion"].detach()  # sg(c_fusion), Eq. (28)
        squared_distances = torch.stack(
            [((output["z"][m] - c_anchor) ** 2).sum(dim=-1) for m in MODALITIES],
            dim=-1,
        )
        L_align = (beta * squared_distances).sum(dim=-1).mean()

        entropy_terms = []
        for m in MODALITIES:
            attention = output["A"][m]
            term = attention * torch.log(attention.clamp_min(cfg.eps))
            entropy_terms.append(term.sum(dim=-1))
        L_att_ent = torch.stack(entropy_terms, dim=-1).mean()  # Eq. (29)
        L_cam = L_align + cfg.lambda_att_ent * L_att_ent  # Eq. (30)

        L_task = L_multi + L_uni
        L_total = L_task + cfg.lambda_atr * L_atr + cfg.lambda_cam * L_cam
        return {
            "L_total": L_total,
            "L_task": L_task,
            "L_multi": L_multi,
            "L_uni": L_uni,
            "L_ety": L_ety,
            "L_route": L_route,
            "L_atr": L_atr,
            "L_align": L_align,
            "L_att_ent": L_att_ent,
            "L_cam": L_cam,
            "epsilon_uni": epsilon_uni,
            "alpha": alpha,
            "beta": beta,
        }
