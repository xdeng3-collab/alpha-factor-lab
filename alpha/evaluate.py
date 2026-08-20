"""Factor evaluation: IC, quantile backtests, turnover, and cost sensitivity.

The cost sensitivity is the part that matters. A long-short quantile portfolio
with a good-looking IC can be entirely consumed by its own turnover, and a
result that does not say at what cost level the edge dies is not a result.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class ICSummary:
    mean: float          # average rank correlation with the forward return
    std: float
    icir: float          # mean / std, annualised
    hit_rate: float      # fraction of periods with IC > 0
    periods: int
    autocorr_1: float    # factor's own persistence: how fast it decays

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioSummary:
    cost_bps: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    turnover: float      # average one-sided fraction of the book replaced

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def information_coefficient(factor: pd.DataFrame, forward: pd.DataFrame) -> pd.Series:
    """Spearman rank correlation per cross-section.

    Rank rather than Pearson: the claim being tested is ordering, not linearity,
    and ranks are unbothered by the fat tails that returns always have.
    """
    aligned_factor, aligned_forward = factor.align(forward, join="inner")
    values = []
    index = []
    for timestamp in aligned_factor.index:
        a = aligned_factor.loc[timestamp]
        b = aligned_forward.loc[timestamp]
        usable = a.notna() & b.notna()
        if usable.sum() < 5:
            continue
        values.append(a[usable].rank().corr(b[usable].rank()))
        index.append(timestamp)
    return pd.Series(values, index=pd.Index(index, name=aligned_factor.index.name))


def summarize_ic(ic: pd.Series, factor: pd.DataFrame | None = None) -> ICSummary:
    clean = ic.dropna()
    if clean.empty:
        raise ValueError("no periods with a computable IC")
    std = float(clean.std())
    autocorr = 0.0
    if factor is not None and len(factor) > 1:
        stacked = factor.stack(future_stack=True) if hasattr(pd.DataFrame, "stack") else factor.stack()
        lagged = factor.shift(1).stack(future_stack=True) if hasattr(pd.DataFrame, "stack") else factor.shift(1).stack()
        joint = pd.concat([stacked, lagged], axis=1).dropna()
        if len(joint) > 2:
            autocorr = float(joint.iloc[:, 0].corr(joint.iloc[:, 1]))

    return ICSummary(
        mean=float(clean.mean()),
        std=std,
        icir=float(clean.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0,
        hit_rate=float((clean > 0).mean()),
        periods=int(len(clean)),
        autocorr_1=autocorr,
    )


def quantile_weights(factor: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """Dollar-neutral long-short weights: long the top bucket, short the bottom."""
    if quantiles < 2:
        raise ValueError("need at least two quantiles")
    weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    for timestamp in factor.index:
        row = factor.loc[timestamp].dropna()
        if len(row) < quantiles * 2:
            continue
        ranked = row.rank(method="first")
        edge = len(row) / quantiles
        longs = ranked[ranked > len(row) - edge].index
        shorts = ranked[ranked <= edge].index
        if len(longs) == 0 or len(shorts) == 0:
            continue
        weights.loc[timestamp, longs] = 0.5 / len(longs)
        weights.loc[timestamp, shorts] = -0.5 / len(shorts)
    return weights


def backtest(
    weights: pd.DataFrame,
    forward: pd.DataFrame,
    *,
    cost_bps: float = 0.0,
) -> tuple[pd.Series, PortfolioSummary]:
    """Period returns and a summary, net of one-sided transaction cost.

    Weights at t earn the return from t to t+1, and the rebalance from the
    previous weights is charged at t. Both halves of that sentence are places a
    backtest commonly cheats.
    """
    aligned_weights, aligned_forward = weights.align(forward, join="inner")
    gross = (aligned_weights * aligned_forward).sum(axis=1, min_count=1)

    traded = aligned_weights.diff().abs().sum(axis=1)
    traded.iloc[0] = aligned_weights.iloc[0].abs().sum()  # putting the book on
    cost = traded * (cost_bps / 10000.0)
    net = (gross - cost).dropna()

    if net.empty:
        raise ValueError("no periods survived alignment")

    annual_return = float(net.mean() * TRADING_DAYS)
    annual_vol = float(net.std() * np.sqrt(TRADING_DAYS))
    curve = (1.0 + net).cumprod()
    drawdown = float((curve / curve.cummax() - 1.0).min())

    return net, PortfolioSummary(
        cost_bps=cost_bps,
        annual_return=annual_return,
        annual_vol=annual_vol,
        sharpe=annual_return / annual_vol if annual_vol > 0 else 0.0,
        max_drawdown=drawdown,
        turnover=float(traded.mean()),
    )


def cost_curve(
    weights: pd.DataFrame,
    forward: pd.DataFrame,
    *,
    levels: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
) -> pd.DataFrame:
    """Sharpe across transaction-cost levels: where does the edge die?"""
    rows = [backtest(weights, forward, cost_bps=level)[1].as_dict() for level in levels]
    return pd.DataFrame(rows)


def breakeven_cost_bps(curve: pd.DataFrame) -> float:
    """Linearly interpolated cost at which Sharpe crosses zero."""
    above = curve[curve["sharpe"] > 0]
    below = curve[curve["sharpe"] <= 0]
    if above.empty:
        return 0.0
    if below.empty:
        return float("inf")
    last_good = above.iloc[-1]
    first_bad = below.iloc[0]
    span = first_bad["sharpe"] - last_good["sharpe"]
    if span == 0:
        return float(last_good["cost_bps"])
    fraction = -last_good["sharpe"] / span
    return float(last_good["cost_bps"] + fraction * (first_bad["cost_bps"] - last_good["cost_bps"]))


def walk_forward(
    factor: pd.DataFrame,
    forward: pd.DataFrame,
    *,
    train: int = 250,
    test: int = 60,
) -> pd.DataFrame:
    """In-sample vs out-of-sample IC over rolling windows.

    Nothing is fitted here, so the comparison isolates one thing: how much of an
    IC is the period rather than the factor. A signal whose out-of-sample IC is
    a fraction of its in-sample IC was describing the past, not predicting.
    """
    rows = []
    index = factor.index
    start = 0
    while start + train + test <= len(index):
        in_slice = slice(start, start + train)
        out_slice = slice(start + train, start + train + test)
        in_ic = information_coefficient(factor.iloc[in_slice], forward.iloc[in_slice])
        out_ic = information_coefficient(factor.iloc[out_slice], forward.iloc[out_slice])
        if not in_ic.empty and not out_ic.empty:
            rows.append(
                {
                    "window_start": index[start],
                    "test_start": index[start + train],
                    "ic_in_sample": float(in_ic.mean()),
                    "ic_out_of_sample": float(out_ic.mean()),
                }
            )
        start += test
    return pd.DataFrame(rows)
