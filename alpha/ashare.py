"""China A-share daily data: download, cache, and turn into a research panel.

Real data brings four problems the synthetic panel never has, and each one is a
place a backtest quietly lies. This module handles them in the data layer, so no
factor has to remember to:

* **Adjustment.** Prices are *back*-adjusted (后复权). Forward adjustment (前复权)
  rescales the whole history every time a new dividend is paid, so a price that
  looks like it was available in 2019 was in fact computed using a 2025
  dividend. That is look-ahead hiding in the data vendor rather than the code.
* **Untradable days.** A stock that is suspended, or closed at its daily price
  limit, could not have been bought or sold at that close. Its factor value is
  masked out for that day rather than letting the backtest trade it.
* **New listings.** The first weeks after an IPO run without normal limits and
  carry returns no factor is meant to explain. The first ``min_listed_days``
  sessions of every stock are masked.
* **Survivorship.** The ``all`` universe is every currently listed A-share
  *plus* every delisted one the exchanges still publish. Index universes
  (``csi300``, ``csi500``) use today's constituents and are therefore biased
  towards stocks that did well enough to be in the index today; they are there
  because they download in minutes, and the CLI says so whenever they are used.

Volume is the traded amount in yuan (成交额), which, unlike share volume, does
not jump at a stock split. Float market cap is recovered point-in-time from the
same row as amount / turnover rate, which is what lets the CLI neutralise
factors against size without a second data source.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .panel import Panel

COLUMNS = {
    "日期": "date",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交额": "amount",
    "涨跌幅": "pct_change",
    "换手率": "turnover_rate",
}

# ChiNext moved from 10% to 20% limits on this date (registration-system reform).
CHINEXT_20PCT_FROM = pd.Timestamp("2020-08-24")


@dataclass(frozen=True)
class AshareData:
    """A panel plus the two things a real market adds to it."""

    panel: Panel
    tradable: pd.DataFrame  # bool: could this stock be traded at this close?
    float_cap: pd.DataFrame  # yuan, from the same row's amount / turnover rate

    def mask(self, factor: pd.DataFrame) -> pd.DataFrame:
        """Blank out factor values on days the stock could not be traded."""
        return factor.where(self.tradable)


# --------------------------------------------------------------------------
# price limits


def limit_pct(code: str, date: pd.Timestamp) -> float:
    """Daily price limit in percent for a board, on a date.

    ST stocks trade with a 5% limit, but ST status history is not in this data
    source, so an ST stock pinned at 5% is not detected as limit-bound. The
    README says so; it is the known gap in the tradability mask.
    """
    if code.startswith("688"):
        return 20.0
    if code.startswith(("300", "301")):
        return 20.0 if date >= CHINEXT_20PCT_FROM else 10.0
    if code.startswith(("4", "8", "92")):
        return 30.0
    return 10.0


def at_limit(code: str, frame: pd.DataFrame) -> pd.Series:
    """True where the close sat at the up or down limit.

    The reported percent change is rounded to two decimals and limit prices are
    rounded to the cent, so "at the limit" means within 0.1 percentage points
    of it *and* closing at the day's extreme in that direction.
    """
    limits = pd.Series([limit_pct(code, d) for d in frame.index], index=frame.index)
    near = frame["pct_change"].abs() >= limits - 0.1
    up = near & (frame["pct_change"] > 0) & (frame["close"] >= frame["high"])
    down = near & (frame["pct_change"] < 0) & (frame["close"] <= frame["low"])
    return up | down


# --------------------------------------------------------------------------
# building the panel (pure, tested offline)


def build(
    frames: dict[str, pd.DataFrame],
    *,
    start: str | None = None,
    end: str | None = None,
    min_listed_days: int = 60,
) -> AshareData:
    """Stack per-stock daily frames into wide, aligned research frames.

    Each frame is indexed by date with the columns in ``COLUMNS``' values.
    Days a stock is absent from (suspension, before listing, after delisting)
    are NaN in the panel and False in ``tradable``. Nothing is forward-filled:
    a price carried across a suspension is a price nobody could trade at.
    """
    if not frames:
        raise ValueError("no stock data to build a panel from")

    close, amount, cap, ok = {}, {}, {}, {}
    for code, frame in frames.items():
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        valid = frame["close"].gt(0) & frame["amount"].gt(0)
        frame = frame[valid]
        if frame.empty:
            continue
        close[code] = frame["close"]
        amount[code] = frame["amount"]
        cap[code] = frame["amount"] / (frame["turnover_rate"] / 100.0).replace(0.0, np.nan)
        seasoned = pd.Series(np.arange(len(frame)) >= min_listed_days, index=frame.index)
        ok[code] = seasoned & ~at_limit(code, frame)

    close_df = pd.DataFrame(close).sort_index()
    if start is not None:
        close_df = close_df.loc[pd.Timestamp(start):]
    if end is not None:
        close_df = close_df.loc[:pd.Timestamp(end)]
    close_df = close_df.dropna(axis=1, how="all")
    index, columns = close_df.index, close_df.columns

    def wide(parts: dict) -> pd.DataFrame:
        return pd.DataFrame(parts).reindex(index=index, columns=columns)

    tradable = wide(ok).fillna(False).astype(bool) & close_df.notna()
    return AshareData(
        panel=Panel(close=close_df, volume=wide(amount)),
        tradable=tradable,
        float_cap=wide(cap),
    )


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Eastmoney's Chinese column names to ours, indexed by date."""
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"unexpected AkShare schema, missing {sorted(missing)}")
    frame = raw[list(COLUMNS)].rename(columns=COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date").astype(float)


def normalise_sina(raw: pd.DataFrame) -> pd.DataFrame:
    """Sina's frame to ours.

    Sina reports turnover as a fraction rather than a percent and has no
    percent-change column. The change is recomputed from back-adjusted closes,
    which equals the exchange's figure (the limit reference price is itself
    ex-dividend adjusted); across a suspension it is measured from the last
    session traded, which is also the exchange's reference.
    """
    needed = {"date", "close", "high", "low", "amount", "turnover"}
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError(f"unexpected Sina schema, missing {sorted(missing)}")
    frame = raw.assign(date=pd.to_datetime(raw["date"])).set_index("date").sort_index()
    out = frame[["close", "high", "low", "amount"]].astype(float)
    out["pct_change"] = out["close"].pct_change() * 100.0
    out["turnover_rate"] = frame["turnover"].astype(float) * 100.0
    return out[list(COLUMNS.values())[1:]]


def sina_symbol(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh" + code
    if code.startswith(("4", "8", "92")):
        return "bj" + code
    return "sz" + code


# --------------------------------------------------------------------------
# download and cache (network; not exercised by the test suite)


def universe(name: str) -> pd.DataFrame:
    """Codes to download, with a ``source`` column saying where each came from."""
    import akshare as ak

    if name in ("csi300", "csi500"):
        index_code = {"csi300": "000300", "csi500": "000905"}[name]
        cons = ak.index_stock_cons_csindex(symbol=index_code)
        return pd.DataFrame({"code": cons["成分券代码"].astype(str).str.zfill(6),
                             "source": f"{name} constituents as of today"})
    if name == "all":
        listed = ak.stock_info_a_code_name()["code"].astype(str).str.zfill(6)
        sh = ak.stock_info_sh_delist()["公司代码"].astype(str).str.zfill(6)
        sz = ak.stock_info_sz_delist()["证券代码"].astype(str).str.zfill(6)
        parts = [pd.DataFrame({"code": listed, "source": "listed"}),
                 pd.DataFrame({"code": sh, "source": "delisted (SSE)"}),
                 pd.DataFrame({"code": sz, "source": "delisted (SZSE)"})]
        return pd.concat(parts).drop_duplicates("code").reset_index(drop=True)
    raise ValueError(f"unknown universe {name!r}; use csi300, csi500 or all")


def fetch(
    codes: Iterable[str],
    cache: Path,
    *,
    start: str,
    end: str,
    pause: float = 0.6,
    retries: int = 5,
    timeout: float = 30.0,
    log: Callable[[str], None] = print,
) -> dict[str, str]:
    """Download back-adjusted daily bars into ``cache/<code>.csv``.

    Sina first, Eastmoney as the fallback: Eastmoney refuses connections from
    some networks outright, and Sina does not serve most delisted stocks, so
    neither alone covers the ``all`` universe.

    Resumable: a code whose file already exists is skipped, so an interrupted
    download picks up where it stopped. Both sources drop connections when
    requests come too fast, so requests are paced and failures back off
    exponentially. Returns the codes that still failed, with the reason.
    """
    import socket

    import akshare as ak

    # AkShare's Sina request sets no timeout, and one stalled connection would
    # otherwise hang the whole download indefinitely. requests falls back to
    # the socket default when it is given none.
    socket.setdefaulttimeout(timeout)

    def from_sina(code: str) -> pd.DataFrame:
        raw = ak.stock_zh_a_daily(symbol=sina_symbol(code), adjust="hfq",
                                  start_date=start.replace("-", ""), end_date=end.replace("-", ""))
        return normalise_sina(raw)

    def from_eastmoney(code: str) -> pd.DataFrame:
        raw = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="hfq",
                                 start_date=start.replace("-", ""), end_date=end.replace("-", ""))
        return normalise(raw)

    cache.mkdir(parents=True, exist_ok=True)
    codes = list(codes)
    failed: dict[str, str] = {}
    for n, code in enumerate(codes, 1):
        path = cache / f"{code}.csv"
        if path.exists():
            continue
        for attempt in range(retries):
            errors = []
            not_served = False
            for source in (from_sina, from_eastmoney):
                try:
                    source(code).to_csv(path)
                    break
                except Exception as error:  # the sources fail in many shapes
                    errors.append(f"{source.__name__}: {type(error).__name__}: {error}"[:150])
                    # Sina answers a code it has no data for (most delisted
                    # stocks) with an unparseable body. That will not change
                    # on retry, so one Eastmoney attempt is all it gets.
                    not_served = not_served or (source is from_sina and isinstance(error, ValueError))
            if path.exists():
                failed.pop(code, None)
                break
            failed[code] = " | ".join(errors)
            if not_served:
                break
            time.sleep(pause * 2 ** (attempt + 1) + random.uniform(0, 1))
        time.sleep(pause + random.uniform(0, pause / 2))
        if n % 100 == 0 or n == len(codes):
            log(f"  {n}/{len(codes)} done, {len(failed)} failing")
    return failed


def write_manifest(cache: Path, **fields) -> None:
    fields["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (cache / "manifest.json").write_text(json.dumps(fields, indent=2, ensure_ascii=False))


def load(cache: Path, **kwargs) -> tuple[AshareData, dict]:
    """Read every cached stock and build the research frames."""
    manifest_path = cache / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{cache} has no manifest.json; run `python cli.py fetch` first")
    manifest = json.loads(manifest_path.read_text())
    frames = {}
    for path in sorted(cache.glob("[0-9]*.csv")):
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(frame):
            frames[path.stem] = frame
    return build(frames, **kwargs), manifest
