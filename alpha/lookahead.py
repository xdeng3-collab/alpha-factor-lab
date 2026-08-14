"""Empirical look-ahead detection.

The `Panel.asof` interface makes look-ahead structurally impossible for factors
that use it. This module handles the rest: any factor, however it was written,
can be checked by truncating the data and seeing whether its historical values
move.

The property is simple and absolute. If a factor's value at time t is computed
only from information available at t, then deleting every row after t cannot
change it. If the value changes, the factor read the future. There is no
tolerance band and no judgement call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .panel import Panel

FactorFn = Callable[[Panel], pd.DataFrame]


@dataclass(frozen=True)
class LookaheadReport:
    clean: bool
    checked_timestamps: int
    first_violation: object | None = None
    max_drift: float = 0.0
    worst_instrument: str | None = None

    def __str__(self) -> str:
        if self.clean:
            return f"clean: {self.checked_timestamps} cut points, no value changed"
        return (
            f"LOOK-AHEAD at {self.first_violation}: value drifted by "
            f"{self.max_drift:.6g} (worst instrument {self.worst_instrument})"
        )


def detect(
    factor: FactorFn,
    panel: Panel,
    *,
    cut_points: int = 12,
    warmup: int = 60,
    tolerance: float = 1e-9,
) -> LookaheadReport:
    """Recompute ``factor`` on truncated panels and compare the overlap.

    ``tolerance`` absorbs floating-point non-associativity only. It is not a
    budget for "a little bit of look-ahead": anything a real future-peek does
    will exceed it by orders of magnitude.
    """
    full = factor(panel)
    if not isinstance(full, pd.DataFrame):
        raise TypeError("a factor must return a DataFrame indexed like the panel")

    timestamps = panel.timestamps
    if len(timestamps) <= warmup + 2:
        raise ValueError("panel is too short to test with this warmup")

    candidates = np.linspace(warmup, len(timestamps) - 2, cut_points, dtype=int)
    max_drift = 0.0
    worst_instrument = None
    first_violation = None

    for position in sorted(set(candidates.tolist())):
        cutoff = timestamps[position]
        truncated = factor(panel.truncate_after(cutoff))

        shared_rows = full.index.intersection(truncated.index)
        shared_cols = full.columns.intersection(truncated.columns)
        a = full.loc[shared_rows, shared_cols]
        b = truncated.loc[shared_rows, shared_cols]

        # NaN in the same place in both is agreement, not a difference.
        both_nan = a.isna() & b.isna()
        drift = (a - b).abs().where(~both_nan, 0.0)
        # A value that exists in one and not the other is a maximal difference.
        drift = drift.mask(a.isna() ^ b.isna(), np.inf)

        if drift.size == 0:
            continue
        worst = float(np.nanmax(drift.to_numpy()))
        if worst > max_drift:
            max_drift = worst
            worst_instrument = str(drift.max(axis=0).idxmax())
        if worst > tolerance and first_violation is None:
            first_violation = cutoff

    return LookaheadReport(
        clean=first_violation is None,
        checked_timestamps=len(set(candidates.tolist())),
        first_violation=first_violation,
        max_drift=max_drift,
        worst_instrument=worst_instrument,
    )


def assert_clean(factor: FactorFn, panel: Panel, **kwargs) -> None:
    """Raise unless the factor is free of look-ahead. For use in test suites."""
    report = detect(factor, panel, **kwargs)
    if not report.clean:
        raise AssertionError(str(report))
