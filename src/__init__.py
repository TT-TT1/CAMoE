"""Reproducible CAMoE implementation matching the manuscript notation."""

from .config import CAMoEConfig
from .losses import CAMoELoss
from .metrics import sentiment_metrics
from .model import CAMoE

__all__ = ["CAMoE", "CAMoEConfig", "CAMoELoss", "sentiment_metrics"]
