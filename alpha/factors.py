"""Factor definitions and the cross-sectional pipeline.

Each factor takes a Panel and returns a frame of raw scores aligned to the
panel's index. None of them look past their own row, and the test suite proves
that rather than asserting it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .panel import Panel

FactorFn = Callable[[Panel], pd.DataFrame]
REGISTRY: dict[str, FactorFn] = {}


def register(name: str) -> Callable[[FactorFn], FactorFn]:
    def wrap(fn: FactorFn) -> FactorFn:
        REGISTRY[name] = fn
        return fn
    return wrap


@register("reversal_1")
def reversal_1(panel: Panel) -> pd.DataFrame:
    """Negative of the most recent return. Buy what just fell."""
    return -panel.close.pct_change(1)


@register("reversal_5")
def reversal_5(panel: Panel) -> pd.DataFrame:
    return -panel.close.pct_change(5)


@register("momentum_20")
def momentum_20(panel: Panel) -> pd.DataFrame:
    return panel.close.pct_change(20)


@register("momentum_60_skip_5")
def momentum_60_skip_5(panel: Panel) -> pd.DataFrame:
    """Medium-term momentum excluding the last week.

    The skip is the point: the most recent days carry short-term reversal, which
    pulls against momentum and muddies the signal.
    """
    return panel.close.shift(5) / panel.close.shift(60) - 1.0


@register("volatility_20")
def volatility_20(panel: Panel) -> pd.DataFrame:
    """Negative realised volatility: the low-vol tilt."""
    return -panel.close.pct_change().rolling(20).std()


@register("volume_trend")
def volume_trend(panel: Panel) -> pd.DataFrame:
    if panel.volume is None:
        raise ValueError("volume_trend needs volume data")
    short = panel.volume.rolling(5).mean()
    long = panel.volume.rolling(60).mean()
    return np.log(short / long)


@register("price_volume_corr")
def price_volume_corr(panel: Panel) -> pd.DataFrame:
    """Rolling correlation of returns and volume changes, negated.

    A falling price on rising volume is conviction; on thin volume it is noise.
    """
    if panel.volume is None:
        raise ValueError("price_volume_corr needs volume data")
    returns = panel.close.pct_change()
    volume_change = np.log(panel.volume).diff()
    return -returns.rolling(20).corr(volume_change)


def winsorize(frame: pd.DataFrame, limit: float = 3.0) -> pd.DataFrame:
    """Clip each cross-section to +/- ``limit`` median absolute deviations.

    MAD rather than standard deviation, because the outliers being clipped are
    exactly what inflates a standard deviation and lets them survive.
    """
    median = frame.median(axis=1)
    mad = (frame.sub(median, axis=0)).abs().median(axis=1)
    scale = mad.replace(0.0, np.nan) * 1.4826  # MAD to sigma for a normal
    lower = median - limit * scale
    upper = median + limit * scale
    return frame.clip(lower=lower, upper=upper, axis=0)


def standardize(frame: pd.DataFrame) -> pd.DataFrame:
    """Zero-mean, unit-variance within each cross-section."""
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def neutralize(frame: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """Residualise the factor against one exposure, cross-section by cross-section.

    Without this step a "momentum factor" is frequently just a sector bet. The
    IC usually falls after neutralising, and that drop is information: it is the
    part of the signal that was never stock selection to begin with.
    """
    if exposure.shape != frame.shape:
        raise ValueError("exposure must match the factor's shape")

    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for timestamp in frame.index:
        y = frame.loc[timestamp]
        x = exposure.loc[timestamp]
        usable = y.notna() & x.notna()
        if usable.sum() < 3:
            continue
        yv = y[usable].to_numpy(dtype=float)
        xv = x[usable].to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(xv), xv])
        coefficients, *_ = np.linalg.lstsq(design, yv, rcond=None)
        out.loc[timestamp, usable[usable].index] = yv - design @ coefficients
    return out


def prepare(frame: pd.DataFrame, *, exposure: pd.DataFrame | None = None) -> pd.DataFrame:
    """Winsorize, optionally neutralize, then standardize. Order matters.

    Clipping before neutralising stops a single outlier from dragging the
    regression line; standardising last puts every factor on one scale so they
    can be combined.
    """
    out = winsorize(frame)
    if exposure is not None:
        out = neutralize(out, exposure)
    return standardize(out)
