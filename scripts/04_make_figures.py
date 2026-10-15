"""Draw every figure used in the README into results/figures/."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

from loadforecast import config
from loadforecast import features as F
from loadforecast.plotting import AXIS, COLORS, GRID, INK, INK_2, LINE, SERIES, SURFACE, save, setup

OUT = config.FIG_DIR
RES = config.RESULTS_DIR
HOLIDAY_WEEK = ("2019-05-27", "2019-06-02")  # Ascension Day (Thu) + bridge day (Fri)
NORMAL_WEEK = ("2019-11-11", "2019-11-17")
GW = FuncFormatter(lambda v, _: f"{v / 1000:.0f}".replace("-", "\u2212"))
THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}")


def subtitle(ax, text):
    """A line of secondary text between the title and the plot (title needs pad >= 24)."""
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def legend_top(ax, ncol, **kw):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.07), ncol=ncol, borderaxespad=0, columnspacing=1.4, **kw)


def fig_overview(ds):
    fig, ax = plt.subplots(figsize=(9, 3.1))
    daily = ds.load.mean(axis=1)
    spans = [
        ("train", "Training\n2015 – 2017", None),
        ("val", "Validation\n2018", GRID),
        ("test", "Test\n2019", "#d9d8d0"),
        ("2020", "Stress test\n2020", GRID),
    ]
    for split, label, shade in spans:
        start, end = (pd.Timestamp(s) for s in config.SPLITS[split])
        if shade:
            ax.axvspan(start, end + pd.Timedelta(days=1), color=shade, alpha=0.55, lw=0)
        ax.text(start + (end - start) / 2, 1.0, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                color=INK_2, fontsize=8.5)
    ax.plot(ds.dates, daily, color=INK, lw=0.7)
    ax.yaxis.set_major_formatter(GW)
    ax.set_ylabel("Mean load of the day (GW)")
    ax.set_xlim(ds.dates[0], ds.dates[-1])
    ax.set_title("German electricity load, daily average", pad=34)
    save(fig, OUT / "01_load_overview.png")


def fig_profiles(ds):
    days = ds.split("train", "val")
    cal = ds.cal.iloc[days]
    load = ds.load[days]
    special = (cal["holiday"] >= 0.5).to_numpy() | (cal["xmas"] > 0).to_numpy()
    groups = {
        "Monday – Friday": (cal["dow"] < 5).to_numpy() & ~special,
        "Saturday": (cal["dow"] == 5).to_numpy() & ~special,
        "Sunday or public holiday": (cal["dow"] == 6).to_numpy() | special,
    }
    seasons = {"Winter (Dec – Feb)": [12, 1, 2], "Summer (Jun – Aug)": [6, 7, 8]}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, (season, months) in zip(axes, seasons.items()):
        in_season = np.isin(ds.dates[days].month, months)
        for color, (name, mask) in zip(SERIES, groups.items()):
            ax.plot(range(24), load[mask & in_season].mean(axis=0), color=color, label=name)
        ax.set_title(season, fontsize=10)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("Hour of the day (local time)")
    axes[0].yaxis.set_major_formatter(GW)
    axes[0].set_ylabel("Average load (GW)")
    axes[0].set_ylim(bottom=30_000)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.76, wspace=0.12)
    fig.suptitle("Daily load profiles, 2015 – 2018", x=0.08, y=0.99, ha="left", va="top", fontsize=11,
                 fontweight="bold", color=INK)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.075, 0.925), ncol=3, columnspacing=1.4)
    save(fig, OUT / "02_daily_profiles.png")


def fig_acf(ds):
    days = ds.split("train", "val")
    x = F.flat(ds.load[days[0] : days[-1] + 1])
    x = x - x.mean()
    lags = np.arange(0, 24 * 15 + 1)
    acf = np.array([1.0] + [x[k:] @ x[:-k] / (x @ x) for k in lags[1:]])
    fig, ax = plt.subplots(figsize=(9, 2.9))
    ax.plot(lags, acf, color=COLORS["Ensemble"])
    for k, label in [(24, "1 day"), (168, "1 week"), (336, "2 weeks")]:
        ax.plot([k], [acf[k]], "o", ms=5, color=COLORS["Ensemble"], mec=SURFACE, mew=1.5)
        ax.annotate(f"{label}\n{acf[k]:.2f}", (k, acf[k]), xytext=(0, 8), textcoords="offset points", ha="center",
                    va="bottom", color=INK_2, fontsize=8.5)
    ax.set_xticks(np.arange(0, 24 * 15 + 1, 24 * 2))
    ax.set_xlim(0, lags[-1])
    ax.set_ylim(-0.6, 1.25)
    ax.set_yticks([-0.5, 0, 0.5, 1])
    ax.axhline(0, color=AXIS, lw=0.6)
    ax.set_xlabel("Lag (hours)")
    ax.set_title("Autocorrelation of hourly load, 2015 – 2018")
    save(fig, OUT / "03_autocorrelation.png")


def fig_sarima():
    params = pd.read_csv(RES / "sarima_parameters.csv", index_col="hour")
    acf = pd.read_csv(RES / "sarima_residual_acf.csv", index_col="hour")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    effects = {
        "Public holiday (Mon–Fri)": "holiday_weekday",
        "Christmas break (Mon–Fri)": "xmas_weekday",
        "Bridge day": "bridge",
    }
    for color, (label, col) in zip(SERIES, effects.items()):
        ax1.plot(params.index, params[col], color=color, label=label)
    ax1.axhline(0, color=AXIS, lw=0.6)
    ax1.yaxis.set_major_locator(MultipleLocator(5000))
    ax1.yaxis.set_major_formatter(GW)
    ax1.set_xticks([0, 6, 12, 18, 23])
    ax1.set_xlabel("Hour of the day")
    ax1.set_ylabel("Effect on load (GW)")
    ax1.set_title("Estimated calendar effects", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8.5)

    hour = 12
    band = 1.96 / np.sqrt(params.loc[hour, "n_resid"])  # approximate 95% band for white noise
    lags = np.arange(1, acf.shape[1] + 1)
    ax2.axhspan(-band, band, color=GRID, alpha=0.8, lw=0)
    ax2.bar(lags, acf.loc[hour], width=0.5, color=COLORS["SARIMA"])
    ax2.axhline(0, color=AXIS, lw=0.6)
    ax2.set_xticks([1, 7, 14, 21, 28])
    ax2.set_xlabel("Lag (days)")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title(f"Residual autocorrelation, {hour}:00 model", fontsize=10)
    fig.suptitle("SARIMA: what the 24 hourly models learned (fit on 2015 – 2018)", x=0.07, ha="left", fontsize=11,
                 fontweight="bold", color=INK, y=1.02)
    fig.subplots_adjust(wspace=0.3)
    save(fig, OUT / "04_sarima.png")


def fig_example_weeks(pred, interval):
    test = pred[pred.split == "test"].reset_index(drop=True)
    test = pd.concat([test, interval], axis=1)
    test["time"] = test["date"] + pd.to_timedelta(test["hour"], unit="h")
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharey=True)
    for ax, (start, end), title in [
        (axes[0], NORMAL_WEEK, "An ordinary week (11 – 17 November 2019)"),
        (axes[1], HOLIDAY_WEEK, "Ascension week (27 May – 2 June 2019): Thursday holiday, Friday bridge day"),
    ]:
        w = test[(test.date >= start) & (test.date <= end)]
        ax.fill_between(w.time, w.lower, w.upper, color=COLORS["Ensemble"], alpha=0.12, lw=0, label="Ensemble 90% interval")
        ax.plot(w.time, w["actual"], color=INK, lw=LINE + 0.3, label="Actual load")
        ax.plot(w.time, w["TSO"], color=COLORS["TSO"], label="TSO official forecast")
        ax.plot(w.time, w["Ensemble"], color=COLORS["Ensemble"], label="Ensemble forecast")
        ticks = pd.date_range(start, end, freq="D")
        ax.set_xticks(ticks + pd.Timedelta(hours=12), [d.strftime("%a %d") for d in ticks])
        ax.set_xticks(ticks, minor=True)
        ax.tick_params(axis="x", which="major", length=0)
        ax.tick_params(axis="x", which="minor", length=4, color=AXIS)
        ax.grid(True, which="minor", axis="x", color=GRID, lw=0.6)
        ax.set_xlim(ticks[0], ticks[-1] + pd.Timedelta(days=1))
        ax.yaxis.set_major_formatter(GW)
        ax.set_ylabel("Load (GW)")
        ax.set_title(title, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 3, 0, 2]
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left", bbox_to_anchor=(0, 1.13),
                   ncol=4, borderaxespad=0)
    fig.subplots_adjust(hspace=0.45)
    save(fig, OUT / "05_example_weeks.png")


def fig_mae_by_hour():
    by_hour = pd.read_csv(RES / "mae_by_hour.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for name in ["TSO", "SARIMA", "GBM", "MLP", "LSTM", "Ensemble"]:
        lw = LINE + 0.6 if name in ("Ensemble", "TSO") else LINE
        ax.plot(by_hour.index, by_hour[name], color=COLORS[name], lw=lw, label=name, zorder=3 if name == "Ensemble" else 2)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 23])
    ax.set_xlabel("Hour of the target day (lead time = hour + 15 h)")
    ax.set_ylabel("MAE (MW)")
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set_ylim(bottom=0)
    ax.set_title("Error by hour of the day, test year 2019", pad=30)
    legend_top(ax, ncol=6)
    save(fig, OUT / "06_mae_by_hour.png")


def fig_day_types():
    by_type = pd.read_csv(RES / "mae_by_day_type.csv", index_col=0)
    order = ["Weekday", "Weekend", "Holiday or bridge day", "Christmas break"]
    by_type = by_type.loc[order]
    models = ["Seasonal naive", "TSO", "Ensemble"]
    fig, ax = plt.subplots(figsize=(9, 3.3))
    width = 0.24
    x = np.arange(len(order))
    for i, m in enumerate(models):
        ax.bar(x + (i - 1) * (width + 0.02), by_type[m], width=width, color=COLORS[m], label=m)
    ax.set_xticks(x, [f"{t}\n({n} days)" for t, n in zip(order, by_type["days"])])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("MAE (MW)")
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.set_title("Error by type of target day, test year 2019", pad=30)
    legend_top(ax, ncol=3)
    save(fig, OUT / "07_mae_by_day_type.png")


def fig_coverage():
    cov = pd.read_csv(RES / "ensemble_coverage_by_hour.csv", index_col=0)["coverage"]
    fig, ax = plt.subplots(figsize=(9, 2.9))
    ax.bar(cov.index, cov, width=0.6, color=COLORS["Ensemble"])
    ax.axhline(0.9, color=INK_2, lw=0.9)
    ax.text(23.6, 0.9, "target 90%", va="center", ha="left", color=INK_2, fontsize=8.5)
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 23])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Hour of the target day")
    ax.set_ylabel("Share of 2019 inside interval")
    ax.set_title("Coverage of the 90% conformal intervals, by hour")
    save(fig, OUT / "08_interval_coverage.png")


def fig_stress_test(pred):
    s = pred[pred.split == "2020"]
    ape = s[["Seasonal naive", "TSO", "Ensemble"]].sub(s["actual"], axis=0).abs().div(s["actual"], axis=0) * 100
    daily = ape.groupby(s["date"]).mean().rolling(7, center=True, min_periods=4).mean()
    fig, ax = plt.subplots(figsize=(9, 3.3))
    lockdown = (pd.Timestamp("2020-03-16"), pd.Timestamp("2020-05-04"))
    ax.axvspan(*lockdown, color=GRID, alpha=0.7, lw=0)
    ax.text(lockdown[0] + (lockdown[1] - lockdown[0]) / 2, 0.97, "first lockdown", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=INK_2, fontsize=8.5)
    for m in ["Seasonal naive", "TSO", "Ensemble"]:
        ax.plot(daily.index, daily[m], color=COLORS[m], label=m, lw=LINE + (0.5 if m == "Ensemble" else 0))
    ax.set_ylabel("Daily MAPE, 7-day mean (%)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(daily.index[0], daily.index[-1])
    months = pd.date_range("2020-01-01", "2020-09-01", freq="MS")
    ax.set_xticks(months, [m.strftime("%b") for m in months])
    ax.set_title("Stress test: January – September 2020 (models refit on 2015 – 2019)", pad=30)
    legend_top(ax, ncol=3)
    save(fig, OUT / "09_stress_test_2020.png")


def fig_learning_curves():
    curves = json.loads((RES / "learning_curves.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    for ax, name in zip(axes, ["MLP", "LSTM"]):
        for i, hist in enumerate(curves[name]):
            val = np.array(hist["val_mae"])
            ax.plot(np.arange(1, len(val) + 1), val, color=COLORS[name], lw=1.0, alpha=0.9,
                    label="validation MAE, one line per seed" if i == 0 else None)
            b = hist["best_epoch"]
            ax.plot([b], [val[b - 1]], "o", ms=5, color=COLORS[name], mec=SURFACE, mew=1.5,
                    label="restored epoch (early stopping)" if i == 0 else None)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("MAE (standardised units)")
    fig.suptitle("Validation curves of the tuned networks (fit on 2015 – 2017)", x=0.07, ha="left", fontsize=11,
                 fontweight="bold", color=INK, y=1.02)
    save(fig, OUT / "10_learning_curves.png")


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = F.load_dataset()
    pred = pd.read_csv(RES / "predictions.csv", parse_dates=["date"])
    interval = pd.read_csv(RES / "ensemble_interval_90_test.csv")
    fig_overview(ds)
    fig_profiles(ds)
    fig_acf(ds)
    fig_sarima()
    fig_example_weeks(pred, interval)
    fig_mae_by_hour()
    fig_day_types()
    fig_coverage()
    fig_stress_test(pred)
    fig_learning_curves()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
