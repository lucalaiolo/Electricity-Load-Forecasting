"""Download the OPSD hourly data and turn it into a (days x 24) matrix in local time."""

from __future__ import annotations

import hashlib
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path = config.RAW_CSV, url: str = config.OPSD_URL, retries: int = 5) -> Path:
    """Download the OPSD csv (~130 MB) once and verify its checksum."""
    if dest.exists() and sha256(dest) == config.OPSD_SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            break
        except OSError as err:  # connection resets, timeouts
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"download failed ({err}); retrying in {wait}s")
            time.sleep(wait)
    digest = sha256(tmp)
    if digest != config.OPSD_SHA256:
        raise RuntimeError(f"checksum mismatch for {url}: got {digest}")
    tmp.replace(dest)
    return dest


def hourly_to_daily(s: pd.Series, tz: str = config.TIMEZONE) -> pd.DataFrame:
    """Reshape an hourly UTC series into a (days x 24) frame indexed by local date.

    Daylight-saving days are made regular: the hour that does not exist on the
    23-hour spring day is linearly interpolated, and the doubled 02:00 hour of
    the 25-hour autumn day is averaged.
    """
    local = s.tz_convert(tz)
    frame = pd.DataFrame(
        {"date": local.index.tz_localize(None).normalize(), "hour": local.index.hour, "y": local.to_numpy()}
    )
    daily = frame.pivot_table(index="date", columns="hour", values="y", aggfunc="mean", dropna=False)
    daily = daily.reindex(columns=range(config.HOURS))
    daily = daily.interpolate(axis=1, limit_area="inside")
    daily.columns = [f"h{h:02d}" for h in daily.columns]
    daily.index.name = "date"
    return daily


def build_processed(raw: Path = config.RAW_CSV, out: Path = config.PROCESSED_CSV) -> pd.DataFrame:
    """Actual load and TSO forecast as one long-by-day table, saved to ``out``."""
    df = pd.read_csv(raw, usecols=["utc_timestamp", config.TARGET_COL, config.TSO_COL])
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    df = df.set_index("utc_timestamp")
    load = hourly_to_daily(df[config.TARGET_COL])
    tso = hourly_to_daily(df[config.TSO_COL])
    days = pd.date_range(config.FIRST_DAY, config.DATA_END, freq="D", name="date")
    load, tso = load.reindex(days), tso.reindex(days)
    if load.isna().any().any():
        raise ValueError("missing actual load values in the study period")
    table = pd.concat({"load": load, "tso": tso}, axis=1)
    table.columns = [f"{a}_{b}" for a, b in table.columns]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, float_format="%.1f")
    return table


def load_processed(path: Path = config.PROCESSED_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (load, tso) as (days x 24) frames with a DatetimeIndex of local dates."""
    table = pd.read_csv(path, index_col="date", parse_dates=["date"])
    load = table.filter(like="load_").rename(columns=lambda c: c.removeprefix("load_"))
    tso = table.filter(like="tso_").rename(columns=lambda c: c.removeprefix("tso_"))
    return load, tso


def flat(daily: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Concatenate the days of a (days x 24) matrix into one regular hourly series."""
    return np.asarray(daily, dtype=float).reshape(-1)
