"""Reference forecasts that need no training."""

import numpy as np

from ..features import Dataset


def seasonal_naive(ds: Dataset, days: np.ndarray, weeks: int = 1) -> np.ndarray:
    """Same hours of the same weekday ``weeks`` weeks ago (averaged if weeks > 1)."""
    return np.mean([ds.load[days - 7 * w] for w in range(1, weeks + 1)], axis=0)


def tso(ds: Dataset, days: np.ndarray) -> np.ndarray:
    """Official day-ahead forecast published by the German TSOs on ENTSO-E."""
    return ds.tso[days]
