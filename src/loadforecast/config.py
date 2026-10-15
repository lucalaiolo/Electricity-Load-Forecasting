"""Project-wide constants: paths, data source, forecasting protocol and splits."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "time_series_60min_singleindex.csv"
PROCESSED_CSV = DATA_DIR / "processed" / "de_load_daily.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Open Power System Data, "Time series" package, version 2020-10-06 (hourly, single index).
OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)
OPSD_SHA256 = "6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8"

TARGET_COL = "DE_load_actual_entsoe_transparency"  # actual total load, MW
TSO_COL = "DE_load_forecast_entsoe_transparency"  # official day-ahead forecast, MW
TIMEZONE = "Europe/Berlin"

# Forecasting protocol: the forecast for day D is issued at 10:00 local time on D-1,
# the latest time the TSOs may publish their day-ahead forecast (2 h before the
# 12:00 gate closure of the day-ahead auction). Load of hours 00-09 of D-1 is known.
ISSUE_HOUR = 10
HOURS = 24
LOOKBACK_HOURS = 168  # one week of hourly history for the window-based models

# Chronological splits of the target days (inclusive, local calendar days).
FIRST_DAY = "2015-01-02"  # 00:00 of 1 Jan 2015 is missing in the source
SPLITS = {
    "train": ("2015-01-02", "2017-12-31"),  # fit
    "val": ("2018-01-01", "2018-12-31"),  # choose hyper-parameters
    "test": ("2019-01-01", "2019-12-31"),  # final evaluation, touched once
    "2020": ("2020-01-01", "2020-09-30"),  # stress test: COVID-19 lockdowns
}
DATA_END = SPLITS["2020"][1]  # last complete day in the OPSD release

SEED = 42
