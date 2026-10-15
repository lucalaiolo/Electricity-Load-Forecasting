"""Turn the (days x 24) load matrix into supervised learning problems.

Every sample is a target day D. Its forecast is issued at ``ISSUE_HOUR`` on D-1, so
the only load values a model may use are those before that moment: all days up to
D-2 and hours 00 .. ISSUE_HOUR-1 of D-1. Calendar information is known in advance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .calendar_features import calendar_table
from .data import flat, load_processed

H = config.HOURS
MAX_LAG_DAYS = 14  # oldest day any model looks at (same hour two weeks earlier)
FLAG_COLS = ["holiday", "bridge", "xmas"]
CONTEXT_LAGS = (1, 2, 7)  # days whose calendar flags explain the lagged load we use


@dataclass
class Dataset:
    dates: pd.DatetimeIndex  # one entry per day
    load: np.ndarray  # (n_days, 24) actual load in MW
    tso: np.ndarray  # (n_days, 24) official TSO day-ahead forecast in MW
    cal: pd.DataFrame  # calendar features per day

    @property
    def series(self) -> np.ndarray:
        return flat(self.load)

    def days(self, start: str, end: str) -> np.ndarray:
        """Indices of target days in [start, end] that have enough history."""
        mask = (self.dates >= start) & (self.dates <= end)
        idx = np.flatnonzero(mask)
        return idx[idx >= MAX_LAG_DAYS]

    def split(self, *names: str) -> np.ndarray:
        """Target days of one or more consecutive splits, e.g. split("train", "val")."""
        return np.concatenate([self.days(*config.SPLITS[name]) for name in names])


def load_dataset() -> Dataset:
    load, tso = load_processed()
    return Dataset(load.index, load.to_numpy(), tso.to_numpy(), calendar_table(load.index))


def cutoff(d: np.ndarray | int) -> np.ndarray | int:
    """Position in the flat hourly series of the first value NOT known when forecasting day d."""
    return (np.asarray(d) - 1) * H + config.ISSUE_HOUR


def windows(series: np.ndarray, days: np.ndarray, length: int = config.LOOKBACK_HOURS) -> np.ndarray:
    """(n, length) matrix of the last ``length`` hourly values known at issue time."""
    ends = cutoff(days)
    return np.stack([series[e - length : e] for e in ends])


def calendar_block(ds: Dataset, days: np.ndarray) -> pd.DataFrame:
    """Calendar of the target day plus holiday flags of the days the lags come from."""
    cal = ds.cal
    target = cal.iloc[days].drop(columns="dow").reset_index(drop=True)
    dow = pd.get_dummies(cal["dow"].iloc[days]).reindex(columns=range(7), fill_value=False)
    dow = dow.astype(float).reset_index(drop=True).add_prefix("dow_")
    context = [
        cal[FLAG_COLS].iloc[days - lag].reset_index(drop=True).add_suffix(f"_lag{lag}") for lag in CONTEXT_LAGS
    ]
    return pd.concat([dow, target, *context], axis=1)


def wide_features(ds: Dataset, days: np.ndarray, length: int = config.LOOKBACK_HOURS) -> pd.DataFrame:
    """Lookback window + calendar, one row per target day (used by the MLP)."""
    win = pd.DataFrame(windows(ds.series, days, length), columns=[f"lag_{length - i}h" for i in range(length)])
    return pd.concat([win, calendar_block(ds, days)], axis=1)


def long_features(ds: Dataset, days: np.ndarray) -> pd.DataFrame:
    """One row per (target day, hour) with hand-crafted lags (used by gradient boosting)."""
    Y, cal = ds.load, ds.cal
    n = len(days)
    d = np.repeat(days, H)
    hour = np.tile(np.arange(H), n)
    observed = slice(0, config.ISSUE_HOUR)  # hours of D-1 known at issue time
    yday = Y[days - 1, observed].mean(axis=1)
    week_on_week = np.repeat(yday / Y[days - 8, observed].mean(axis=1), H)

    feats = {
        "hour": hour,
        "dow": cal["dow"].to_numpy()[d],
        "doy": ds.dates.dayofyear.to_numpy()[d],
        **{c: cal[c].to_numpy()[d] for c in FLAG_COLS},
        # same hour 1 day ago is only known for the morning hours; NaN otherwise
        "same_hour_1d": np.where(hour < config.ISSUE_HOUR, Y[d - 1, hour], np.nan),
        **{f"same_hour_{k}d": Y[d - k, hour] for k in (2, 3, 4, 5, 6, 7, 14)},
        "last_value": np.repeat(Y[days - 1, config.ISSUE_HOUR - 1], H),
        "yday_morning_mean": np.repeat(yday, H),
        "week_on_week": week_on_week,  # this morning vs the same morning a week ago
        "naive_scaled": Y[d - 7, hour] * week_on_week,
        "mean_last_7d": np.repeat(Y[days[:, None] - np.arange(2, 9)].mean(axis=(1, 2)), H),
    }
    for lag in (1, 2, 7, 14):
        for c in FLAG_COLS:
            feats[f"{c}_lag{lag}"] = cal[c].to_numpy()[d - lag]
    return pd.DataFrame(feats)


def sequence_features(
    ds: Dataset, days: np.ndarray, mean: float, std: float, length: int = config.LOOKBACK_HOURS
) -> np.ndarray:
    """(n, length, 11) per-hour inputs for the LSTM: scaled load, hour of day, weekday, holiday."""
    ends = cutoff(days)
    pos = ends[:, None] - length + np.arange(length)[None, :]  # flat positions in the window
    day_of_pos, hour_of_pos = pos // H, pos % H
    load = (ds.series[pos] - mean) / std
    angle = 2 * np.pi * hour_of_pos / H
    dow = np.eye(7)[ds.cal["dow"].to_numpy()[day_of_pos]]
    hol = ds.cal["holiday"].to_numpy()[day_of_pos]
    return np.concatenate(
        [load[..., None], np.sin(angle)[..., None], np.cos(angle)[..., None], dow, hol[..., None]], axis=-1
    ).astype(np.float32)


def targets(ds: Dataset, days: np.ndarray) -> np.ndarray:
    return ds.load[days]
