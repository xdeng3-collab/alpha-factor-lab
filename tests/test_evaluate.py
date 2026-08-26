import numpy as np
import pandas as pd
import pytest

from alpha import evaluate as ev
from alpha import factors
from alpha.panel import synthetic_panel


def test_pipeline_reports_nothing_on_pure_noise():
    # The single most important property. A pipeline that finds signal in noise
    # will find signal anywhere, and no amount of real-data testing reveals it.
    panel = synthetic_panel(50, 600, seed=7, embed_signal=0.0)
    forward = panel.forward_return(1)
    prepared = factors.prepare(factors.reversal_1(panel))
    summary = ev.summarize_ic(ev.information_coefficient(prepared, forward))
    assert abs(summary.mean) < 0.03
    assert 0.42 < summary.hit_rate < 0.58


def test_a_planted_signal_is_recovered():
    panel = synthetic_panel(50, 600, seed=7, embed_signal=0.3)
    prepared = factors.prepare(factors.reversal_1(panel))
    summary = ev.summarize_ic(ev.information_coefficient(prepared, panel.forward_return(1)))
    assert summary.mean > 0.1


def test_costs_only_ever_reduce_returns():
    panel = synthetic_panel(40, 400, seed=3, embed_signal=0.2)
    weights = ev.quantile_weights(factors.prepare(factors.reversal_1(panel)))
    curve = ev.cost_curve(weights, panel.forward_return(1))
    assert curve["sharpe"].is_monotonic_decreasing
    assert curve["turnover"].nunique() == 1  # cost does not change what is traded


def test_breakeven_is_where_sharpe_crosses_zero():
    curve = pd.DataFrame({"cost_bps": [0.0, 10.0, 20.0], "sharpe": [2.0, 1.0, -1.0]})
    assert ev.breakeven_cost_bps(curve) == pytest.approx(15.0)
    assert ev.breakeven_cost_bps(pd.DataFrame({"cost_bps": [0.0], "sharpe": [1.0]})) == np.inf
    assert ev.breakeven_cost_bps(pd.DataFrame({"cost_bps": [0.0], "sharpe": [-1.0]})) == 0.0


def test_quantile_weights_are_dollar_neutral():
    panel = synthetic_panel(40, 200, seed=5)
    weights = ev.quantile_weights(factors.prepare(factors.momentum_20(panel)))
    active = weights.loc[weights.abs().sum(axis=1) > 0]
    assert len(active) > 50
    assert active.sum(axis=1).abs().max() < 1e-9
    assert np.allclose(active.abs().sum(axis=1), 1.0)


def test_putting_the_book_on_is_charged_for():
    # A backtest that starts fully invested for free has stolen its first
    # period of cost. The build must appear in the traded volume.
    panel = synthetic_panel(30, 200, seed=11, embed_signal=0.2)
    weights = ev.quantile_weights(factors.prepare(factors.reversal_1(panel)))
    forward = panel.forward_return(1)

    first_active = weights.index[weights.abs().sum(axis=1) > 0][0]
    free, _ = ev.backtest(weights, forward, cost_bps=0.0)
    charged, _ = ev.backtest(weights, forward, cost_bps=100.0)
    assert charged.loc[first_active] < free.loc[first_active]
    # And over the whole path, cost strictly reduces the cumulative return.
    assert charged.sum() < free.sum()


def test_neutralization_removes_the_exposure_it_is_given():
    panel = synthetic_panel(40, 300, seed=13)
    exposure = panel.close.pct_change().rolling(60).std()
    raw = factors.winsorize(factors.momentum_20(panel))
    residual = factors.neutralize(raw, exposure)

    correlations = []
    for timestamp in residual.index[-80:]:
        a, b = residual.loc[timestamp], exposure.loc[timestamp]
        usable = a.notna() & b.notna()
        if usable.sum() > 10:
            correlations.append(a[usable].corr(b[usable]))
    assert np.nanmax(np.abs(correlations)) < 1e-8


def test_winsorize_clips_outliers_and_leaves_most_of_the_bulk_alone():
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 60)))
    frame.iloc[:, 0] = 500.0  # one instrument with an absurd value every period
    clipped = factors.winsorize(frame)

    # The outlier is pulled back to the cross-sectional bound.
    assert clipped.iloc[:, 0].max() < 5.0

    # A 3-MAD clip on normal data touches the extreme tail and nothing else.
    # Asserting that *nothing* in the bulk moves would be wrong: the tail of
    # 1,180 normal draws genuinely reaches past three sigma.
    bulk = clipped.iloc[:, 1:] - frame.iloc[:, 1:]
    touched = (bulk.abs() > 1e-12).to_numpy().mean()
    assert touched < 0.02


def test_standardize_gives_each_cross_section_unit_scale():
    frame = pd.DataFrame(np.random.default_rng(1).normal(10, 5, size=(30, 40)))
    out = factors.standardize(frame)
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(out.std(axis=1), 1.0, atol=1e-12)
