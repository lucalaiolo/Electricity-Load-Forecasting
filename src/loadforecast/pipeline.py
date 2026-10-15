"""Model specifications: how each model is tuned on the validation year and refit.

Protocol (identical for every model):
  1. fit on the training years, choose hyper-parameters by MAE on the validation year;
  2. refit on training + validation years with the chosen hyper-parameters;
  3. forecast the test year once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import config
from . import features as F
from .metrics import mae
from .models.gbm import GBM
from .models.neural import NeuralForecaster
from .models.sarima import HourlySARIMA, calendar_exog

NN_SEEDS = (0, 1, 2, 3, 4)


@dataclass
class Tuned:
    params: dict
    val_pred: np.ndarray  # validation-year forecasts of the tuned model (fit on train only)
    log: list[dict] = field(default_factory=list)  # one row per configuration tried


class SARIMAModel:
    """24 regressions with SARIMA errors (one per hour), weekly seasonality.

    The hours already observed at issue time are forecast one day ahead in their daily
    series, the others two days ahead. The order is chosen separately for these two
    horizons, by validation MAE, from a small grid of classic specifications.
    """

    name = "SARIMA"
    grid = [
        ((0, 1, 1), (0, 1, 1, 7)),  # the "airline" model
        ((1, 1, 1), (0, 1, 1, 7)),
        ((1, 0, 1), (0, 1, 1, 7)),
        ((1, 0, 1), (1, 1, 1, 7)),
        ((2, 0, 0), (0, 1, 1, 7)),
    ]

    def _fit(self, ds, last_day: int, orders: dict) -> HourlySARIMA:
        """Fit on the whole history up to and including `last_day` (the first days included)."""
        X = calendar_exog(ds.cal)
        return HourlySARIMA(orders).fit(ds.load[: last_day + 1], X[: last_day + 1])

    def tune(self, ds, train, val) -> Tuned:
        X, Y_val = calendar_exog(ds.cal), F.targets(ds, val)
        preds, log = {}, []
        for cfg in self.grid:
            model = self._fit(ds, train[-1], {1: cfg, 2: cfg})
            preds[cfg] = model.predict(ds.load, X, val)
            hourly_mae = np.abs(preds[cfg] - Y_val).mean(axis=0)
            log.append({
                "order": cfg[0], "seasonal_order": cfg[1], "val_mae_1_step": hourly_mae[: config.ISSUE_HOUR].mean(),
                "val_mae_2_step": hourly_mae[config.ISSUE_HOUR :].mean(), "val_mae": hourly_mae.mean(),
                "aic_sum": sum(r.aic for r in model.results),
            })  # fmt: skip
        best = {
            1: min(self.grid, key=lambda c: log[self.grid.index(c)]["val_mae_1_step"]),
            2: min(self.grid, key=lambda c: log[self.grid.index(c)]["val_mae_2_step"]),
        }
        val_pred = np.where(np.arange(config.HOURS) < config.ISSUE_HOUR, preds[best[1]], preds[best[2]])
        return Tuned({"orders": best}, val_pred, log)

    def fit(self, ds, days, params) -> SARIMAModel:
        self.model = self._fit(ds, days[-1], {int(k): v for k, v in params["orders"].items()})
        return self

    def predict(self, ds, days) -> np.ndarray:
        return self.model.predict(ds.load, calendar_exog(ds.cal), days)


class GBMModel:
    name = "GBM"
    grid = [
        {"max_leaf_nodes": leaves, "min_samples_leaf": leaf_size} for leaves in (15, 31) for leaf_size in (20, 100)
    ]
    learning_rate = 0.1
    max_iter = 3000

    def tune(self, ds, train, val) -> Tuned:
        X_tr, X_val = F.long_features(ds, train), F.long_features(ds, val)
        Y_val = F.targets(ds, val)
        log, best = [], None
        for cfg in self.grid:
            gbm = GBM(learning_rate=self.learning_rate, max_iter=self.max_iter, **cfg)
            gbm.fit(X_tr, F.targets(ds, train))
            curve = [mae(Y_val, p) for p in gbm.staged_predict(X_val)]
            n_iter = int(np.argmin(curve)) + 1
            log.append({**cfg, "n_iter": n_iter, "val_mae": curve[n_iter - 1]})
            if best is None or curve[n_iter - 1] < best[0]:
                best = (curve[n_iter - 1], {**cfg, "max_iter": n_iter}, gbm, n_iter)
        _, params, gbm, n_iter = best
        val_pred = next(p for i, p in enumerate(gbm.staged_predict(X_val), 1) if i == n_iter)
        return Tuned({**params, "learning_rate": self.learning_rate}, val_pred, log)

    def fit(self, ds, days, params) -> GBMModel:
        self.model = GBM(**params).fit(F.long_features(ds, days), F.targets(ds, days))
        return self

    def predict(self, ds, days) -> np.ndarray:
        return self.model.predict(F.long_features(ds, days))


class _NeuralModel:
    """Shared tuning logic: early stopping on the validation year picks the epoch count,
    which is reused (median over seeds) when refitting on train + validation."""

    kind: str
    grid: list[dict]
    train_kw = {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 32}

    def inputs(self, ds, days) -> tuple[np.ndarray, ...]:
        raise NotImplementedError

    def prepare(self, ds, days) -> None:
        """Hook for statistics that must come from the fitting days only."""

    def tune(self, ds, train, val) -> Tuned:
        torch.set_num_threads(4)
        self.prepare(ds, train)
        X_tr, X_val = self.inputs(ds, train), self.inputs(ds, val)
        Y_tr, Y_val = F.targets(ds, train), F.targets(ds, val)
        log, best = [], None
        for arch in self.grid:
            net = NeuralForecaster(self.kind, seeds=NN_SEEDS, arch=arch, epochs=400, patience=40, **self.train_kw)
            net.fit(X_tr, Y_tr, X_val, Y_val)
            pred = net.predict(X_val)
            score = mae(Y_val, pred)
            epochs = int(np.median(net.best_epochs))
            log.append({**arch, "epochs": epochs, "val_mae": score})
            if best is None or score < best[0]:
                best = (score, {"arch": arch, "epochs": epochs}, pred, net)
        self.tuned_net = best[3]  # kept for the learning-curve figure
        return Tuned(best[1], best[2], log)

    def fit(self, ds, days, params) -> _NeuralModel:
        torch.set_num_threads(4)
        self.prepare(ds, days)
        self.net = NeuralForecaster(self.kind, seeds=NN_SEEDS, arch=params["arch"], **self.train_kw)
        self.net.fit(self.inputs(ds, days), F.targets(ds, days), epochs=params["epochs"])
        return self

    def predict(self, ds, days) -> np.ndarray:
        return self.net.predict(self.inputs(ds, days))


class MLPModel(_NeuralModel):
    name = "MLP"
    kind = "mlp"
    grid = [
        {"hidden": hidden, "depth": depth, "dropout": 0.2} for depth in (1, 2) for hidden in (128, 256, 512)
    ]

    def inputs(self, ds, days):
        return (F.wide_features(ds, days).to_numpy(),)


class LSTMModel(_NeuralModel):
    name = "LSTM"
    kind = "lstm"
    grid = [{"hidden": hidden, "head": 128, "dropout": 0.1} for hidden in (32, 64)]

    def prepare(self, ds, days):
        self.mean, self.std = float(ds.load[days].mean()), float(ds.load[days].std())

    def inputs(self, ds, days):
        seq = F.sequence_features(ds, days, self.mean, self.std)
        return seq, F.calendar_block(ds, days).to_numpy()


MODELS = [SARIMAModel, GBMModel, MLPModel, LSTMModel]
