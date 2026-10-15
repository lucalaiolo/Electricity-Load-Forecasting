"""Score the forecasts: metrics, error breakdowns, Diebold-Mariano tests, conformal intervals.

Reads results/predictions.csv, writes CSV tables and a markdown summary to results/.
"""

import numpy as np
import pandas as pd

from loadforecast import config, conformal
from loadforecast import features as F
from loadforecast import metrics as M

MODELS = ["Seasonal naive", "TSO", "SARIMA", "GBM", "MLP", "LSTM", "Ensemble"]
ML = ["SARIMA", "GBM", "MLP", "LSTM", "Ensemble"]


def day_type(cal: pd.DataFrame) -> pd.Series:
    kind = np.where(cal["dow"] >= 5, "Weekend", "Weekday")
    special = (cal["holiday"] > 0) | (cal["bridge"] > 0)
    kind = np.where(special, "Holiday or bridge day", kind)
    kind = np.where(cal["xmas"] > 0, "Christmas break", kind)
    return pd.Series(kind, index=cal.index, name="day_type")


def matrix(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy().reshape(-1, config.HOURS)


def main():
    out = config.RESULTS_DIR
    pred = pd.read_csv(out / "predictions.csv", parse_dates=["date"])
    ds = F.load_dataset()
    types = day_type(ds.cal)
    pred["day_type"] = types.reindex(pred["date"]).to_numpy()
    val, test, stress = (pred[pred.split == s].reset_index(drop=True) for s in ("val", "test", "2020"))

    # MASE denominator: in-sample MAE of the weekly naive forecast on the final training span.
    fit_days = ds.split("train", "val")
    scale = M.seasonal_scale(F.flat(ds.load[fit_days[0] : fit_days[-1] + 1]))

    metrics_test = M.summarize(test["actual"].to_numpy(), {m: test[m].to_numpy() for m in MODELS}, scale)
    metrics_val = M.summarize(val["actual"].to_numpy(), {m: val[m].to_numpy() for m in MODELS if m != "TSO"}, scale)
    tso_ok = val["TSO"].notna()  # the TSO forecast is missing for one day of 2018
    metrics_val.loc["TSO"] = M.summarize(val.loc[tso_ok, "actual"].to_numpy(), {"TSO": val.loc[tso_ok, "TSO"].to_numpy()}, scale).iloc[0]
    metrics_val = metrics_val.loc[MODELS]
    metrics_test.to_csv(out / "metrics_test.csv", float_format="%.3f")
    metrics_val.to_csv(out / "metrics_val.csv", float_format="%.3f")

    abs_err = test[MODELS].sub(test["actual"], axis=0).abs()
    by_hour = abs_err.groupby(test["hour"]).mean()
    by_month = abs_err.div(test["actual"], axis=0).groupby(test["date"].dt.month).mean() * 100
    by_type = abs_err.groupby(test["day_type"]).mean()
    by_type.insert(0, "days", test.groupby("day_type").size() // config.HOURS)
    by_hour.to_csv(out / "mae_by_hour.csv", float_format="%.1f")
    by_month.to_csv(out / "mape_by_month.csv", float_format="%.3f")
    by_type.to_csv(out / "mae_by_day_type.csv", float_format="%.1f")

    # Correlation of the hourly errors: low correlation is what makes averaging pay off.
    err = test[MODELS].sub(test["actual"], axis=0)
    err.corr().to_csv(out / "error_correlation.csv", float_format="%.3f")

    # Diebold-Mariano on daily MAE: entry (row, col) is the p-value of
    # "col is more accurate than row".
    daily = abs_err.groupby(test["date"]).mean()
    dm_p = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
    dm_stat = dm_p.copy()
    for a in MODELS:
        for b in MODELS:
            if a != b:
                dm_stat.loc[a, b], dm_p.loc[a, b] = M.diebold_mariano(daily[a].to_numpy(), daily[b].to_numpy())
    dm_p.to_csv(out / "dm_pvalues.csv", float_format="%.4g")
    dm_stat.to_csv(out / "dm_statistics.csv", float_format="%.3f")

    # Split conformal intervals: per-hour quantiles of |validation residuals|, applied to 2019.
    rows = []
    for model in ML:
        resid_val = matrix(val, "actual") - matrix(val, model)
        for alpha in (0.2, 0.1, 0.05):
            lo, hi = conformal.interval(matrix(test, model), resid_val, alpha)
            y = matrix(test, "actual")
            inside = (y >= lo) & (y <= hi)
            row = {"model": model, "nominal": 1 - alpha, "coverage": inside.mean(), "mean_width_MW": (hi - lo).mean()}
            for kind, mask in test.groupby("day_type").groups.items():
                row[f"coverage: {kind}"] = inside.ravel()[mask].mean()
            rows.append(row)
            if model == "Ensemble" and alpha == 0.1:
                pd.DataFrame({"lower": lo.ravel(), "upper": hi.ravel()}).to_csv(
                    out / "ensemble_interval_90_test.csv", index=False, float_format="%.1f"
                )
                pd.Series(inside.mean(axis=0), name="coverage").rename_axis("hour").to_csv(
                    out / "ensemble_coverage_by_hour.csv", float_format="%.4f"
                )
    conf = pd.DataFrame(rows)
    conf.to_csv(out / "conformal.csv", index=False, float_format="%.4f")

    # 2020 stress test (models refit on 2015-2019).
    abs_err20 = stress[MODELS].sub(stress["actual"], axis=0).abs()
    mape20 = abs_err20.div(stress["actual"], axis=0).groupby(stress["date"].dt.month).mean() * 100
    mape20.loc["Jan-Sep"] = abs_err20.div(stress["actual"], axis=0).mean() * 100
    mape20.to_csv(out / "mape_2020_by_month.csv", float_format="%.3f")

    lines = ["# Results summary (generated by scripts/03_evaluate.py)", ""]
    lines += ["## Test year 2019", "", metrics_test.round(3).to_markdown(), ""]
    lines += ["## Validation year 2018 (models fit on 2015-2017)", "", metrics_val.round(3).to_markdown(), ""]
    lines += ["## MAE (MW) by day type, 2019", "", by_type.round(0).to_markdown(), ""]
    lines += ["## Correlation of hourly errors, 2019", "", err.corr().round(2).to_markdown(), ""]
    lines += ["## Diebold-Mariano one-sided p-values (column better than row), daily MAE", "", dm_p.round(4).to_markdown(), ""]
    lines += ["## Split conformal intervals on 2019 (calibrated on 2018)", "", conf.round(3).to_markdown(index=False), ""]
    lines += ["## MAPE (%) by month in 2020 (models refit on 2015-2019)", "", mape20.round(2).to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
