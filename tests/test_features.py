import copy

import numpy as np
import pandas as pd

from loadforecast import config
from loadforecast import features as F
from loadforecast.data import hourly_to_daily


def test_window_ends_at_issue_time(synthetic_dataset):
    ds = synthetic_dataset
    d = np.array([30])
    win = F.windows(ds.series, d)[0]
    assert len(win) == config.LOOKBACK_HOURS
    assert win[-1] == ds.load[29, config.ISSUE_HOUR - 1]  # 09:00 of D-1 is the newest value
    assert win[0] == ds.load[22, config.ISSUE_HOUR]  # 10:00 of D-8 is the oldest one


def test_features_do_not_peek_after_issue_time(synthetic_dataset):
    """Overwriting everything from 10:00 of D-1 onwards must not change the inputs for day D."""
    ds = synthetic_dataset
    d = np.array([40])
    future = copy.deepcopy(ds)
    flat = future.load.reshape(-1)  # view: edits go through to future.load
    flat[F.cutoff(40) :] = np.random.default_rng(1).normal(0, 1e6, len(flat) - F.cutoff(40))

    pd.testing.assert_frame_equal(F.wide_features(ds, d), F.wide_features(future, d))
    pd.testing.assert_frame_equal(F.long_features(ds, d), F.long_features(future, d))
    np.testing.assert_array_equal(
        F.sequence_features(ds, d, 5e4, 1e4), F.sequence_features(future, d, 5e4, 1e4)
    )


def test_daylight_saving_days_have_24_hours():
    idx = pd.date_range("2019-03-30 00:00", "2019-03-31 23:00", freq="h", tz="Europe/Berlin", nonexistent="NaT")
    idx = idx[~idx.isna()].tz_convert("UTC")  # 47 real hours: 02:00 on 31 March does not exist
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    daily = hourly_to_daily(s)
    assert daily.shape == (2, 24)
    assert daily.loc["2019-03-31", "h02"] == (daily.loc["2019-03-31", "h01"] + daily.loc["2019-03-31", "h03"]) / 2

    idx = pd.date_range("2019-10-26 22:00", "2019-10-27 22:00", freq="h", tz="UTC")  # the 25-hour local day
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    daily = hourly_to_daily(s)
    assert daily.shape == (1, 24)
    assert daily.loc["2019-10-27", "h02"] == 2.5  # mean of the two 02:00 hours (values 2 and 3)
    assert daily.loc["2019-10-27", "h03"] == 4.0
