import pandas as pd
import pytest

from loadforecast.calendar_features import calendar_table


def test_holiday_shares():
    cal = calendar_table(pd.date_range("2019-05-29", "2019-06-21", freq="D"))
    assert cal.loc["2019-05-30", "holiday"] == pytest.approx(1.0)  # Ascension Day, national
    assert cal.loc["2019-05-31", "bridge"] == pytest.approx(1.0)  # Friday after it
    assert cal.loc["2019-06-20", "holiday"] == pytest.approx(0.644, abs=1e-3)  # Corpus Christi, 6 states
    assert cal.loc["2019-06-21", "bridge"] == pytest.approx(0.644, abs=1e-3)
    assert cal.loc["2019-06-03", ["holiday", "bridge", "xmas"]].sum() == 0


def test_christmas_break():
    cal = calendar_table(pd.date_range("2019-12-20", "2020-01-05", freq="D"))
    assert cal["xmas"].sum() == 10  # 24 Dec .. 2 Jan
