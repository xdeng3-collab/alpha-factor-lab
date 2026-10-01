"""The A-share data layer, tested offline on hand-built frames."""

import numpy as np
import pandas as pd
import pytest

from alpha import ashare, factors
from alpha.lookahead import detect


def bars(n=80, *, start="2021-01-04", pct=None, seed=0):
    """Daily bars shaped like a normalised AkShare frame."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    pct = rng.normal(0, 1.5, n) if pct is None else np.asarray(pct, dtype=float)
    close = 10 * np.cumprod(1 + pct / 100)
    return pd.DataFrame(
        {
            "close": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "amount": rng.uniform(1e7, 5e7, n),
            "pct_change": pct,
            "turnover_rate": rng.uniform(0.5, 3.0, n),
        },
        index=dates,
    )


def test_limits_follow_board_and_chinext_reform():
    before, after = pd.Timestamp("2020-08-21"), pd.Timestamp("2020-08-24")
    assert ashare.limit_pct("600519", after) == 10.0
    assert ashare.limit_pct("000001", after) == 10.0
    assert ashare.limit_pct("688981", before) == 20.0
    assert ashare.limit_pct("300750", before) == 10.0
    assert ashare.limit_pct("300750", after) == 20.0
    assert ashare.limit_pct("830799", after) == 30.0


def test_a_close_at_the_limit_is_untradable_but_a_big_move_is_not():
    frame = bars(80)
    limit_day, big_day = frame.index[70], frame.index[71]
    frame.loc[limit_day, ["pct_change"]] = 10.0
    frame.loc[limit_day, "high"] = frame.loc[limit_day, "close"]  # closed at the high
    frame.loc[big_day, ["pct_change"]] = 9.0
    frame.loc[big_day, "high"] = frame.loc[big_day, "close"]

    data = ashare.build({"600000": frame}, min_listed_days=0)
    assert not data.tradable.loc[limit_day, "600000"]
    assert data.tradable.loc[big_day, "600000"]


def test_chinext_ten_percent_is_only_a_limit_before_the_reform():
    frame = bars(80, start="2020-06-01")
    early = frame.index[5]
    late = frame.index[-1]
    assert late >= ashare.CHINEXT_20PCT_FROM > early
    for day in (early, late):
        frame.loc[day, "pct_change"] = 10.0
        frame.loc[day, "high"] = frame.loc[day, "close"]
    data = ashare.build({"300001": frame}, min_listed_days=0)
    assert not data.tradable.loc[early, "300001"]
    assert data.tradable.loc[late, "300001"]


def test_suspensions_are_gaps_not_forward_filled_prices():
    a = bars(80, seed=1)
    b = bars(80, seed=2).drop(index=a.index[40:45])  # suspended for a week
    data = ashare.build({"600000": a, "000002": b}, min_listed_days=0)
    gap = a.index[40:45]
    assert data.panel.close.loc[gap, "000002"].isna().all()
    assert not data.tradable.loc[gap, "000002"].any()
    assert data.tradable.loc[gap, "600000"].all()
    # No return is invented across the gap either.
    assert pd.isna(data.panel.forward_return(1).loc[a.index[39], "000002"])


def test_new_listings_are_masked_for_their_first_sessions():
    old = bars(80, seed=3)
    new = bars(40, start=str(old.index[40].date()), seed=4)
    data = ashare.build({"600000": old, "601999": new}, min_listed_days=10)
    listed = data.panel.close["601999"].first_valid_index()
    first = data.tradable["601999"].loc[listed:].iloc[:10]
    assert not first.any()
    assert data.tradable["601999"].loc[listed:].iloc[10:].all()


def test_float_cap_comes_from_amount_over_turnover():
    frame = bars(80)
    data = ashare.build({"600000": frame}, min_listed_days=0)
    expected = frame["amount"] / (frame["turnover_rate"] / 100)
    pd.testing.assert_series_equal(data.float_cap["600000"], expected, check_names=False)


def test_mask_blanks_untradable_days_only():
    frame = bars(80)
    data = ashare.build({"600000": frame}, min_listed_days=20)
    masked = data.mask(data.panel.close)
    assert masked["600000"].iloc[:20].isna().all()
    assert masked["600000"].iloc[20:].notna().all()


def test_date_window_and_schema_checks():
    data = ashare.build({"600000": bars(80)}, start="2021-02-01", end="2021-03-01", min_listed_days=0)
    assert data.panel.timestamps[0] >= pd.Timestamp("2021-02-01")
    assert data.panel.timestamps[-1] <= pd.Timestamp("2021-03-01")
    with pytest.raises(ValueError, match="schema"):
        ashare.normalise(pd.DataFrame({"日期": ["2021-01-04"], "收盘": [1.0]}))
    with pytest.raises(ValueError):
        ashare.build({})


def test_factors_stay_clean_on_a_gappy_real_shaped_panel():
    frames = {f"60{i:04d}": bars(200, seed=i) for i in range(12)}
    frames["600005"] = frames["600005"].drop(index=frames["600005"].index[90:110])
    panel = ashare.build(frames, min_listed_days=0).panel
    for name, fn in factors.REGISTRY.items():
        assert detect(fn, panel, cut_points=6, warmup=80).clean, name


def test_sina_frames_are_put_on_the_same_footing():
    raw = pd.DataFrame({
        "date": ["2021-01-05", "2021-01-04", "2021-01-06"],
        "close": [11.0, 10.0, 12.1], "high": [11.0, 10.2, 12.1], "low": [10.1, 9.8, 11.0],
        "amount": [2e7, 1e7, 3e7], "turnover": [0.02, 0.01, 0.03],
        "volume": [1, 1, 1], "open": [1, 1, 1], "outstanding_share": [1, 1, 1],
    })
    frame = ashare.normalise_sina(raw)
    assert list(frame.index) == sorted(frame.index)
    assert frame["pct_change"].iloc[1:].round(6).tolist() == [10.0, 10.0]
    assert frame["turnover_rate"].tolist() == [1.0, 2.0, 3.0]  # percent, like Eastmoney
    data = ashare.build({"600000": frame}, min_listed_days=0)
    assert not data.tradable["600000"].iloc[1:].any()  # both rises closed limit-up


def test_sina_symbols_carry_the_exchange():
    assert ashare.sina_symbol("600519") == "sh600519"
    assert ashare.sina_symbol("000002") == "sz000002"
    assert ashare.sina_symbol("300750") == "sz300750"
    assert ashare.sina_symbol("830799") == "bj830799"
