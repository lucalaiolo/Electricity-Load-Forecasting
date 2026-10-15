import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

from loadforecast import config
from loadforecast.calendar_features import calendar_table
from loadforecast.models.sarima import HourlySARIMA, calendar_exog, fit_series, forecast_series

ORDER, SEASONAL = (1, 0, 1), (0, 1, 1, 7)


@pytest.fixture(scope="module")
def weekly_series():
    """AR(1) noise around a weekly pattern, with a holiday-like dummy."""
    rng = np.random.default_rng(0)
    n = 300
    X = (rng.random((n, 1)) < 0.05).astype(float)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.8 * u[t - 1] + rng.normal()
    y = 50 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) - 8 * X[:, 0] + u
    params = fit_series(y[:250], X[:250], ORDER, SEASONAL).params
    return y, X, params


@pytest.mark.parametrize("steps", [1, 2])
def test_filter_forecasts_match_statsmodels(weekly_series, steps):
    """The vectorised forecasts equal statsmodels' own forecasts from each origin."""
    y, X, params = weekly_series
    targets = np.array([200, 251, 299])
    ours = forecast_series(y, X, ORDER, SEASONAL, params, targets, steps)
    for target, value in zip(targets, ours):
        origin = target - steps + 1  # first unobserved day
        model = SARIMAX(y[:origin], exog=X[:origin], order=ORDER, seasonal_order=SEASONAL)
        reference = model.filter(params).forecast(steps, exog=X[origin : target + 1])[-1]
        assert value == pytest.approx(reference, rel=1e-9)


def test_hourly_forecasts_ignore_data_after_issue_time(synthetic_dataset):
    """Load observed from 10:00 on D-1 onwards must not change the SARIMA forecast for day D."""
    ds = synthetic_dataset
    X = calendar_exog(ds.cal)
    model = HourlySARIMA({1: ((0, 1, 1), (0, 1, 1, 7)), 2: (ORDER, SEASONAL)}, n_jobs=1).fit(ds.load[:50], X[:50])
    d = np.array([60])
    future = ds.load.copy().reshape(-1)
    cut = (d[0] - 1) * config.HOURS + config.ISSUE_HOUR
    future[cut:] = np.random.default_rng(1).normal(0, 1e6, len(future) - cut)
    np.testing.assert_allclose(
        model.predict(ds.load, X, d), model.predict(future.reshape(-1, config.HOURS), X, d), rtol=1e-12
    )


def test_calendar_exog_splits_holidays_by_weekday():
    cal = calendar_table(pd.date_range("2019-12-23", "2020-01-06", freq="D"))
    X = pd.DataFrame(calendar_exog(cal), index=cal.index)
    assert X.loc["2019-12-25", 0] == 1.0  # Wednesday holiday
    assert X.loc["2019-12-28", 4] == 1.0  # Saturday in the Christmas break
    assert X.loc["2020-01-05"].sum() == 0  # ordinary Sunday
