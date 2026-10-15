"""Calendar features known in advance: weekday, public holidays, Christmas break, season."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

# Population of the 16 German states on 31 Dec 2019 (Destatis), in millions.
# Regional holidays are weighted by the share of the population that has the day off.
STATE_POPULATION = {
    "BW": 11.100, "BY": 13.125, "BE": 3.669, "BB": 2.522, "HB": 0.681, "HH": 1.847,
    "HE": 6.288, "MV": 1.608, "NI": 7.994, "NW": 17.947, "RP": 4.094, "SL": 0.987,
    "SN": 4.072, "ST": 2.195, "SH": 2.904, "TH": 2.133,
}  # fmt: skip


def state_holidays(dates: pd.DatetimeIndex) -> np.ndarray:
    """Boolean matrix (days x states): is the date a public holiday in the state?"""
    years = sorted(set(dates.year))
    columns = []
    for state in STATE_POPULATION:
        cal = holidays.country_holidays("DE", subdiv=state, years=years)
        columns.append([d in cal for d in dates.date])
    return np.array(columns, dtype=bool).T


def calendar_table(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row of calendar features per day.

    holiday : population share with a public holiday (1.0 for national holidays)
    bridge  : population share for which the day is a working day squeezed between a
              holiday and the weekend (Fri after a Thursday holiday, Mon before a
              Tuesday holiday)
    xmas    : Christmas/New Year break, 24 Dec - 2 Jan
    doy_*   : first two Fourier harmonics of the day of the year
    """
    dates = pd.DatetimeIndex(dates)
    one_day = pd.Timedelta(days=1)
    padded = pd.date_range(dates.min() - one_day, dates.max() + one_day, freq="D")
    weights = np.array(list(STATE_POPULATION.values()))
    weights = weights / weights.sum()
    is_hol = state_holidays(padded)
    pos = padded.get_indexer(dates)
    today, before, after = is_hol[pos], is_hol[pos - 1], is_hol[pos + 1]

    dow = dates.dayofweek.to_numpy()
    hol = today @ weights
    bridge = np.where(dow == 4, (before & ~today) @ weights, 0.0)
    bridge += np.where(dow == 0, (after & ~today) @ weights, 0.0)
    xmas = ((dates.month == 12) & (dates.day >= 24)) | ((dates.month == 1) & (dates.day <= 2))
    doy = 2 * np.pi * (dates.dayofyear.to_numpy() - 1) / 365.25

    return pd.DataFrame(
        {
            "dow": dow,
            "holiday": hol,
            "bridge": bridge,
            "xmas": xmas.astype(float),
            "doy_sin1": np.sin(doy),
            "doy_cos1": np.cos(doy),
            "doy_sin2": np.sin(2 * doy),
            "doy_cos2": np.cos(2 * doy),
        },
        index=dates,
    )
