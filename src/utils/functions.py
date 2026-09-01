from __future__ import annotations

import logging
import random
from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


LOGGER = logging.getLogger("EMOE")
_PROBABILITY_EPSILON = 1e-9

__all__ = [
    "dict_to_str",
    "setup_seed",
    "assign_gpu",
    "count_parameters",
    "eva_imp",
    "uni_distill",
    "entropy_balance",
    "orthogonal_loss",
    "cmd_loss",
]


def dict_to_str(src_dict: Mapping[Any, Any]) -> str:
    """Format scalar metrics in insertion order with four decimal places."""

    return "".join(f" {name}: {float(value):.4f} " for name, value in src_dict.items())


def setup_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic cuDNN runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _least_used_gpu(memory_limit: float) -> tuple[int, int]:
    """Return the index and occupied memory of the least-used visible GPU."""

    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError(
            "Automatic GPU selection requires the 'pynvml' package."
        ) from exc

    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            raise RuntimeError("NVML did not report any visible GPU devices.")

        candidates: list[tuple[int, int]] = []
        for device_id in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            used_memory = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            if used_memory < memory_limit:
                candidates.append((device_id, used_memory))

        if not candidates:
            raise RuntimeError(
                f"No GPU has occupied memory below the limit {memory_limit}."
            )
        return min(candidates, key=lambda item: item[1])
    finally:
        pynvml.nvmlShutdown()


def assign_gpu(
    gpu_ids: Sequence[int], memory_limit: float = 1e16
) -> torch.device:
    """Select a requested CUDA device or automatically choose the least-used one.

    An empty mutable sequence is updated with the automatically selected device
    index for compatibility with existing training scripts.  If CUDA is not
    available, the function always returns a CPU device.
    """

    selected_ids = list(gpu_ids)
    if not selected_ids and torch.cuda.is_available():
        device_id, used_memory = _least_used_gpu(memory_limit)
        selected_ids.append(device_id)
        if isinstance(gpu_ids, MutableSequence):
            gpu_ids.append(device_id)
        LOGGER.info("Found gpu %d, used memory %d.", device_id, used_memory)

    if selected_ids and torch.cuda.is_available():
        return torch.device(f"cuda:{int(selected_ids[0])}")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count scalar parameters that participate in gradient optimization."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def eva_imp(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Return element-wise squared prediction errors."""

    return torch.square(y_pred - y_true)


def uni_distill(logits1: Tensor, logits2: Tensor) -> Tensor:
    """Measure mean-squared distance between two softmax distributions."""

    distribution_1 = torch.softmax(logits1, dim=-1)
    distribution_2 = torch.softmax(logits2, dim=-1)
    return torch.square(distribution_1 - distribution_2).mean()


def entropy_balance(probs: Tensor) -> Tensor:
    """Compute the scaled negative-entropy routing regularizer.

    ``probs`` is expected to have shape ``[batch, experts]``.  The historical
    scale factor equal to the number of experts is retained for compatibility.
    """

    if probs.ndim != 2:
        raise ValueError("probs must have shape [batch, experts].")
    safe_probs = probs.clamp_min(_PROBABILITY_EPSILON)
    expert_count = safe_probs.shape[1]
    per_sample = expert_count * (safe_probs * safe_probs.log()).sum(dim=1)
    return per_sample.mean()


def orthogonal_loss(z1: Tensor, z2: Tensor) -> Tensor:
    """Penalize squared cross-correlation between two feature batches."""

    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("z1 and z2 must both have shape [batch, features].")
    if z1.shape[0] != z2.shape[0]:
        raise ValueError("z1 and z2 must have the same batch size.")
    if z1.shape[0] == 0:
        raise ValueError("orthogonal_loss requires a non-empty batch.")

    normalized_1 = F.normalize(z1, p=2, dim=1)
    normalized_2 = F.normalize(z2, p=2, dim=1)
    cross_correlation = torch.einsum("bi,bj->ij", normalized_1, normalized_2)
    cross_correlation = cross_correlation / z1.shape[0]
    return cross_correlation.square().mean()


def cmd_loss(x: Tensor, y: Tensor, K: int = 5) -> Tensor:
    """Compute central moment discrepancy up to moment order ``K``."""

    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("x and y must include batch and feature dimensions.")
    if x.shape[1:] != y.shape[1:]:
        raise ValueError("x and y must have matching non-batch dimensions.")
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("cmd_loss requires non-empty input batches.")
    if not isinstance(K, int) or K < 1:
        raise ValueError("K must be a positive integer.")

    mean_x = x.mean(dim=0)
    mean_y = y.mean(dim=0)
    discrepancy = torch.linalg.vector_norm(mean_x - mean_y, ord=2)

    centered_x = x - mean_x
    centered_y = y - mean_y
    for order in range(2, K + 1):
        moment_x = centered_x.pow(order).mean(dim=0)
        moment_y = centered_y.pow(order).mean(dim=0)
        discrepancy = discrepancy + torch.linalg.vector_norm(
            moment_x - moment_y, ord=2
        )
    return discrepancy
