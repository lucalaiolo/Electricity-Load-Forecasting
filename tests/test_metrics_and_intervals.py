import numpy as np
import pytest

from loadforecast import conformal
from loadforecast import metrics as M


def test_point_metrics():
    y, f = np.array([100.0, 200.0, 400.0]), np.array([110.0, 180.0, 400.0])
    assert M.mae(y, f) == pytest.approx(10.0)
    assert M.rmse(y, f) == pytest.approx(np.sqrt(500 / 3))
    assert M.mape(y, f) == pytest.approx(100 * (0.1 + 0.1 + 0) / 3)
    assert M.bias(y, f) == pytest.approx(-10 / 3)
    assert M.seasonal_scale(np.array([1.0, 2.0, 4.0, 8.0]), season=2) == pytest.approx(4.5)


def test_newey_west_without_lags_is_the_variance():
    x = np.random.default_rng(0).normal(size=500)
    assert M.newey_west_variance(x, 0) == pytest.approx(np.var(x))


def test_diebold_mariano_size_and_power():
    rng = np.random.default_rng(0)
    p_null = [M.diebold_mariano(rng.gamma(2, 1, 365), rng.gamma(2, 1, 365))[1] for _ in range(2000)]
    assert 0.03 < np.mean(np.array(p_null) < 0.05) < 0.07  # about 5% false rejections
    loss_a = rng.gamma(2, 1, 365) + 0.5
    stat, p = M.diebold_mariano(loss_a, rng.gamma(2, 1, 365))
    assert stat > 0 and p < 1e-3  # b clearly better than a


def test_conformal_quantile_rank():
    scores = np.arange(1, 20, dtype=float)[:, None]  # n = 19
    assert conformal.conformal_quantile(scores, 0.1)[0] == 18  # ceil(20 * 0.9) = 18th smallest
    assert np.isinf(conformal.conformal_quantile(scores, 0.01)[0])  # needs more than 19 points


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(500):
        resid = rng.standard_t(3, size=(60, 1))  # calibration residuals
        y_new = rng.standard_t(3, size=(1, 1))
        lo, hi = conformal.interval(np.zeros((1, 1)), resid, alpha=0.1)
        hits.append(conformal.coverage(y_new, lo, hi))
    assert np.mean(hits) >= 0.87  # >= 90% in expectation; 500 draws leave some Monte Carlo slack
