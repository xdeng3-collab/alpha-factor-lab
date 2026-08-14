"""The look-ahead detector is the safety net every other number depends on, so
it is tested from both directions: it must not flag clean factors, and it must
flag leaky ones. A detector that only ever says "clean" passes the first half."""

import pandas as pd
import pytest

from alpha.lookahead import assert_clean, detect
from alpha.panel import synthetic_panel
from alpha import factors


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(25, 400, seed=42)


@pytest.mark.parametrize("name", sorted(factors.REGISTRY))
def test_every_registered_factor_is_free_of_lookahead(panel, name):
    assert_clean(factors.REGISTRY[name], panel, cut_points=6, warmup=80)


@pytest.mark.parametrize("shift", [-1, -3, -20])
def test_a_negative_shift_is_always_caught(panel, shift):
    leaky = lambda p: p.close.pct_change(5).shift(shift)  # noqa: E731
    assert not detect(leaky, panel, cut_points=6, warmup=80).clean


def test_using_the_last_price_as_a_factor_is_caught(panel):
    # The subtler leak: normalising by a statistic of the whole sample. Every
    # historical value moves when more data arrives.
    leaky = lambda p: p.close / p.close.mean()  # noqa: E731
    report = detect(leaky, panel, cut_points=6, warmup=80)
    assert not report.clean
    assert report.max_drift > 0


def test_forward_return_is_the_target_and_is_not_pretending_otherwise(panel):
    # forward_return deliberately contains the future. Confirming that the
    # detector flags it guards against someone using it as an input by mistake.
    assert not detect(lambda p: p.forward_return(1), panel, cut_points=6, warmup=80).clean


def test_asof_cannot_return_the_future(panel):
    cutoff = panel.timestamps[100]
    view = panel.asof(cutoff)
    assert view.timestamps.max() == cutoff
    assert len(view.timestamps) == 101
