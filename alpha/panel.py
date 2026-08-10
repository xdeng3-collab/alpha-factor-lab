"""Point-in-time panel data.

Every look-ahead bug in factor research is the same bug: something that was not
knowable at time t ended up in the factor value at time t. Catching it by
reading code does not work, because the mistakes hide inside a `shift` with the
wrong sign or a `resample` with the wrong label.

So the panel makes the rule structural. `Panel.asof(t)` is the *only* way to
read data, and it cannot return a row after t. A factor written against this
interface is incapable of seeing the future, and `alpha.lookahead` then proves
it empirically for factors that were not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Panel:
    """Wide price/volume frames indexed by timestamp, columns are instruments."""

    close: pd.DataFrame
    volume: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.close.index, pd.Index):
            raise TypeError("close must be indexed by timestamp")
        if not self.close.index.is_monotonic_increasing:
            raise ValueError("close index must be sorted ascending")
        if self.close.index.has_duplicates:
            raise ValueError("close index has duplicate timestamps")
        if self.volume is not None:
            if not self.volume.index.equals(self.close.index):
                raise ValueError("volume must share the close index")
            if list(self.volume.columns) != list(self.close.columns):
                raise ValueError("volume must share the close columns")

    @property
    def instruments(self) -> list[str]:
        return list(self.close.columns)

    @property
    def timestamps(self) -> pd.Index:
        return self.close.index

    def asof(self, timestamp) -> "Panel":
        """Everything knowable at ``timestamp``, inclusive. Nothing after it.

        This is the whole safety property. A factor that only ever calls
        ``asof`` cannot look ahead, because the future rows are not in the
        object it was handed.
        """
        cutoff = self.close.index <= timestamp
        return Panel(
            close=self.close.loc[cutoff],
            volume=None if self.volume is None else self.volume.loc[cutoff],
        )

    def forward_return(self, horizon: int = 1) -> pd.DataFrame:
        """Return from t to t+horizon, aligned to t.

        This is the one frame that deliberately contains the future, because it
        is the prediction target. It is never an input to a factor, and keeping
        it in a separate method from ``asof`` is what stops it becoming one.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        return self.close.shift(-horizon) / self.close - 1.0

    def truncate_after(self, timestamp) -> "Panel":
        """Alias for ``asof``, spelled for use in look-ahead testing."""
        return self.asof(timestamp)


def synthetic_panel(
    n_instruments: int = 60,
    n_periods: int = 750,
    *,
    seed: int = 0,
    embed_signal: float = 0.0,
) -> Panel:
    """A panel with known statistical structure, for testing the machinery.

    ``embed_signal`` injects a genuine one-period reversal: today's return is
    partly the negative of yesterday's. Setting it to zero produces prices with
    no predictable structure at all, which is the more useful case -- an
    evaluation pipeline that reports a healthy IC on pure noise is broken, and
    that is a test worth having.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_periods)
    names = [f"A{i:03d}" for i in range(n_instruments)]

    market = rng.normal(0.0, 0.010, size=n_periods)
    returns = np.zeros((n_periods, n_instruments))
    idiosyncratic = rng.normal(0.0, 0.018, size=(n_periods, n_instruments))
    beta = rng.uniform(0.6, 1.4, size=n_instruments)

    for t in range(n_periods):
        returns[t] = beta * market[t] + idiosyncratic[t]
        if embed_signal and t > 0:
            returns[t] -= embed_signal * returns[t - 1]

    close = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=names)
    volume = pd.DataFrame(
        rng.lognormal(12.0, 0.6, size=(n_periods, n_instruments)), index=dates, columns=names
    )
    return Panel(close=close, volume=volume)
