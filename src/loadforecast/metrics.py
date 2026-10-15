"""Point-forecast metrics and the Diebold-Mariano test."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def mae(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean(np.abs(y - f)))


def rmse(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - f) ** 2)))


def mape(y: np.ndarray, f: np.ndarray) -> float:
    return float(100 * np.mean(np.abs(y - f) / np.abs(y)))


def bias(y: np.ndarray, f: np.ndarray) -> float:
    """Mean error, forecast minus actual: positive means over-forecasting."""
    return float(np.mean(f - y))


def seasonal_scale(series: np.ndarray, season: int = 168) -> float:
    """In-sample MAE of the seasonal naive forecast, the denominator of MASE."""
    return float(np.mean(np.abs(series[season:] - series[:-season])))


def summarize(y: np.ndarray, forecasts: dict[str, np.ndarray], scale: float) -> pd.DataFrame:
    rows = {}
    for name, f in forecasts.items():
        rows[name] = {
            "MAE": mae(y, f),
            "RMSE": rmse(y, f),
            "MAPE": mape(y, f),
            "MASE": mae(y, f) / scale,
            "Bias": bias(y, f),
        }
    return pd.DataFrame(rows).T


def newey_west_variance(x: np.ndarray, lags: int) -> float:
    """Long-run variance of a series with Bartlett weights (Newey & West, 1987)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    lrv = x @ x / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return float(lrv)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """DM test of equal predictive accuracy on a series of (daily) losses.

    d_t = loss_a - loss_b. Returns the statistic and the one-sided p-value of
    H0: E[d] <= 0 against H1: E[d] > 0, i.e. small p-values mean that
    forecast b is significantly more accurate than forecast a.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if lags is None:
        lags = int(math.floor(4 * (n / 100) ** (2 / 9)))  # usual Newey-West rule of thumb
    stat = d.mean() / math.sqrt(newey_west_variance(d, lags) / n)
    p_value = 0.5 * math.erfc(stat / math.sqrt(2))  # 1 - Phi(stat)
    return float(stat), float(p_value)
