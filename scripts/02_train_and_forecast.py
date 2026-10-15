"""Tune every model on 2018, refit on 2015-2018, forecast 2019; refit on 2015-2019, forecast 2020.

Writes to results/:
  predictions.csv        hourly forecasts of every model for the val, test and 2020 splits
  hyperparameters.json   chosen hyper-parameters and the full tuning logs
  learning_curves.json   per-epoch training/validation losses of the tuned networks
  sarima_parameters.csv  estimates and residual diagnostics of the 24 SARIMA models (fit on 2015-2018)
  sarima_residual_acf.csv  autocorrelation of their one-step residuals
"""

import json
import time

import numpy as np
import pandas as pd

from loadforecast import config
from loadforecast import features as F
from loadforecast.models import baselines
from loadforecast.pipeline import MODELS, LSTMModel, MLPModel, SARIMAModel

ML_MODELS = [m.name for m in MODELS]


def long_table(ds, days, preds: dict, split: str) -> pd.DataFrame:
    n = len(days)
    table = pd.DataFrame(
        {
            "date": np.repeat(ds.dates[days].strftime("%Y-%m-%d"), config.HOURS),
            "hour": np.tile(np.arange(config.HOURS), n),
            "split": split,
            "actual": ds.load[days].ravel(),
            "TSO": baselines.tso(ds, days).ravel(),
            "Seasonal naive": baselines.seasonal_naive(ds, days).ravel(),
        }
    )
    for name, pred in preds.items():
        table[name] = pred.ravel()
    table["Ensemble"] = table[ML_MODELS].mean(axis=1)
    return table


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    ds = F.load_dataset()
    train, val, test, stress = ds.split("train"), ds.split("val"), ds.split("test"), ds.split("2020")
    train_val, train_val_test = ds.split("train", "val"), ds.split("train", "val", "test")
    print(f"target days: train {len(train)}, val {len(val)}, test {len(test)}, 2020 {len(stress)}")

    preds = {split: {} for split in ("val", "test", "2020")}
    hyper, curves = {}, {}
    for Model in MODELS:
        t0 = time.time()
        model = Model()
        tuned = model.tune(ds, train, val)
        preds["val"][model.name] = tuned.val_pred
        hyper[model.name] = {"chosen": tuned.params, "grid": tuned.log}
        if isinstance(model, (MLPModel, LSTMModel)):
            curves[model.name] = model.tuned_net.histories

        model.fit(ds, train_val, tuned.params)
        preds["test"][model.name] = model.predict(ds, test)
        if isinstance(model, SARIMAModel):
            model.model.summary().to_csv(config.RESULTS_DIR / "sarima_parameters.csv", float_format="%.5g")
            model.model.residual_acf().to_csv(config.RESULTS_DIR / "sarima_residual_acf.csv", float_format="%.4f")

        model.fit(ds, train_val_test, tuned.params)
        preds["2020"][model.name] = model.predict(ds, stress)
        print(f"{model.name:6s} chosen {tuned.params}  ({time.time() - t0:.0f}s)", flush=True)

    days = {"val": val, "test": test, "2020": stress}
    table = pd.concat([long_table(ds, days[s], preds[s], s) for s in preds], ignore_index=True)
    table.to_csv(config.RESULTS_DIR / "predictions.csv", index=False, float_format="%.1f")
    with open(config.RESULTS_DIR / "hyperparameters.json", "w") as f:
        json.dump(hyper, f, indent=2, default=float)
    with open(config.RESULTS_DIR / "learning_curves.json", "w") as f:
        json.dump(curves, f, default=float)
    print("wrote", config.RESULTS_DIR / "predictions.csv")


if __name__ == "__main__":
    main()
