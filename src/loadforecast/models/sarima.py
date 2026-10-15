"""Seasonal ARIMA with calendar regressors: one model per hour of the day.

Hour h of every day forms a daily series y_d = Y[d, h]. Each series is modelled as a
regression on holiday dummies with SARIMA(p,d,q)(P,D,Q)_7 errors (weekly seasonality)
and fitted by exact Gaussian maximum likelihood (statsmodels' Kalman filter).

At issue time (10:00 on D-1) hour h of D-1 is already known when h < ISSUE_HOUR, so the
forecast for day D is one step ahead in that hour's series; otherwise it is two steps ahead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..config import HOURS, ISSUE_HOUR

EXOG_NAMES = ["holiday_weekday", "holiday_saturday", "bridge", "xmas_weekday", "xmas_weekend"]


def calendar_exog(cal: pd.DataFrame) -> np.ndarray:
    """Holiday regressors. A holiday on a Sunday changes nothing, so effects are split by weekday."""
    weekday, saturday, weekend = cal["dow"] < 5, cal["dow"] == 5, cal["dow"] >= 5
    columns = [cal["holiday"] * weekday, cal["holiday"] * saturday, cal["bridge"], cal["xmas"] * weekday,
               cal["xmas"] * weekend]  # fmt: skip
    return np.column_stack(columns).astype(float)


def steps_ahead(hour: int) -> int:
    return 1 if hour < ISSUE_HOUR else 2


def fit_series(y: np.ndarray, X: np.ndarray, order: tuple, seasonal_order: tuple):
    """Maximum likelihood fit of a regression with SARIMA errors."""
    model = SARIMAX(y, exog=X, order=order, seasonal_order=seasonal_order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # optimiser chatter; convergence is checked by the caller
        return model.fit(disp=False, maxiter=200)


def forecast_series(
    y: np.ndarray, X: np.ndarray, order: tuple, seasonal_order: tuple, params: np.ndarray, days: np.ndarray, steps: int
) -> np.ndarray:
    """Forecasts of y[days] made `steps` days earlier, from one causal Kalman filter pass.

    With a_t = E[state_t | y_<t], the one-step forecast is d_t + Z a_t and the two-step
    forecast of y_t is d_t + Z (c + T a_{t-1}): the filter's prediction for t-1 pushed one
    more step through the transition equation. Values of y at or after t-steps+1 are
    never used, so passing the whole series is safe.
    """
    fr = SARIMAX(y, exog=X, order=order, seasonal_order=seasonal_order).filter(params).filter_results
    if steps == 1:
        return fr.forecasts[0, days]
    T, Z, c = fr.transition[:, :, 0], fr.design[0, :, 0], fr.state_intercept[:, 0]
    a_prev = fr.predicted_state[:, days - 1]
    return fr.obs_intercept[0, days] + Z @ (c[:, None] + T @ a_prev)


class HourlySARIMA:
    """24 regressions with SARIMA errors; `orders` maps steps ahead (1 or 2) to (order, seasonal_order)."""

    def __init__(self, orders: dict[int, tuple], n_jobs: int = 4):
        self.orders, self.n_jobs = orders, n_jobs

    def fit(self, Y: np.ndarray, X: np.ndarray) -> HourlySARIMA:
        """Y: (days, 24) loads of the fitting period, X: (days, k) regressors."""
        jobs = (delayed(fit_series)(Y[:, h], X, *self.orders[steps_ahead(h)]) for h in range(HOURS))
        self.results = Parallel(n_jobs=self.n_jobs)(jobs)
        return self

    def predict(self, Y_all: np.ndarray, X_all: np.ndarray, days: np.ndarray) -> np.ndarray:
        """(len(days), 24) forecasts; Y_all may extend beyond the forecast days (the filter is causal)."""
        cols = []
        for h, res in enumerate(self.results):
            steps = steps_ahead(h)
            cols.append(forecast_series(Y_all[:, h], X_all, *self.orders[steps], res.params, days, steps))
        return np.column_stack(cols)

    def residual_acf(self, nlags: int = 28) -> pd.DataFrame:
        """Autocorrelation of the standardised one-step residuals, lags 1..nlags, one row per hour."""
        rows = []
        for res in self.results:
            resid = res.filter_results.standardized_forecasts_error[0, res.loglikelihood_burn :]
            rows.append(acf(resid, nlags=nlags)[1:])
        return pd.DataFrame(rows, columns=[f"lag{k}" for k in range(1, nlags + 1)]).rename_axis("hour")

    def summary(self) -> pd.DataFrame:
        """Estimated parameters and diagnostics, one row per hour."""
        rows = []
        for h, res in enumerate(self.results):
            order, seasonal_order = self.orders[steps_ahead(h)]
            n_arma = sum(order[::2]) + seasonal_order[0] + seasonal_order[2]
            resid = res.filter_results.standardized_forecasts_error[0, res.loglikelihood_burn :]
            lb = acorr_ljungbox(resid, lags=[14], model_df=n_arma)
            params = dict(zip(res.model.param_names, res.params))
            rows.append({
                "hour": h, "steps_ahead": steps_ahead(h), "order": str(order), "seasonal_order": str(seasonal_order),
                **{EXOG_NAMES[i]: params.pop(f"x{i + 1}") for i in range(len(EXOG_NAMES))}, **params,
                "aic": res.aic, "converged": res.mle_retvals.get("converged"), "n_resid": len(resid),
                "ljung_box_q14": float(lb["lb_stat"].iloc[0]), "ljung_box_p": float(lb["lb_pvalue"].iloc[0]),
            })  # fmt: skip
        return pd.DataFrame(rows).set_index("hour")
