import numpy as np
import pandas as pd
import pytest

from loadforecast.calendar_features import calendar_table
from loadforecast.features import Dataset


@pytest.fixture
def synthetic_dataset() -> Dataset:
    """Ten weeks of hourly load with daily and weekly cycles plus noise."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2019-01-07", periods=70, freq="D")
    hours = np.arange(24)
    daily = 50_000 + 8_000 * np.sin(2 * np.pi * (hours - 6) / 24)
    weekly = np.where(dates.dayofweek >= 5, -7_000, 0)[:, None]
    load = daily[None, :] + weekly + rng.normal(0, 500, (len(dates), 24))
    return Dataset(dates, load, load + rng.normal(0, 800, load.shape), calendar_table(dates))
