"""Split conformal prediction intervals, calibrated separately for each hour of the day."""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Finite-sample corrected (1 - alpha) quantile of nonconformity scores along axis 0.

    With n calibration scores the k-th smallest, k = ceil((n + 1)(1 - alpha)), is
    returned: under exchangeability a new score falls at or below it with
    probability at least 1 - alpha.
    """
    scores = np.sort(np.asarray(scores, dtype=float), axis=0)
    n = scores.shape[0]
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return np.full(scores.shape[1:], np.inf)
    return scores[k - 1]


def interval(forecast: np.ndarray, cal_residuals: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric interval forecast +/- q_h, with q_h computed per hour from |residuals|."""
    q = conformal_quantile(np.abs(cal_residuals), alpha)
    return forecast - q, forecast + q


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y >= lower) & (y <= upper)))
