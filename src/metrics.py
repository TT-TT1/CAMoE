"""ACC7, ACC2, macro-F1, and MAE as defined in manuscript Section 4.2."""

from typing import Dict, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _numpy_1d(values: ArrayLike) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise ValueError("predictions and targets must be finite")
    return result


def _binary_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    class_scores = []
    for label in (0, 1):
        true_positive = np.sum((y_true == label) & (y_pred == label))
        false_positive = np.sum((y_true != label) & (y_pred == label))
        false_negative = np.sum((y_true == label) & (y_pred != label))
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(class_scores))


def sentiment_metrics(predictions: ArrayLike, targets: ArrayLike) -> Dict[str, float]:
    """Return table-ready metrics (accuracies/F1 in percent, MAE unscaled).

    ACC2 and F1 exclude strictly neutral ground-truth samples. F1 is macro-F1,
    matching the manuscript rather than the weighted-F1 used by the old code.
    """

    pred = _numpy_1d(predictions)
    true = _numpy_1d(targets)
    if pred.shape != true.shape or pred.size == 0:
        raise ValueError("predictions and targets must be non-empty and equally sized")

    pred_7 = np.rint(np.clip(pred, -3.0, 3.0)).astype(np.int64)
    true_7 = np.rint(np.clip(true, -3.0, 3.0)).astype(np.int64)
    acc7 = float(np.mean(pred_7 == true_7))

    non_neutral = true != 0.0
    if not np.any(non_neutral):
        raise ValueError("ACC2/F1 are undefined when every target is strictly neutral")
    true_2 = (true[non_neutral] > 0.0).astype(np.int64)
    pred_2 = (pred[non_neutral] > 0.0).astype(np.int64)
    acc2 = float(np.mean(pred_2 == true_2))
    macro_f1 = _binary_macro_f1(true_2, pred_2)

    return {
        "acc7": 100.0 * acc7,
        "acc2": 100.0 * acc2,
        "f1": 100.0 * macro_f1,
        "mae": float(np.mean(np.abs(pred - true))),
    }
