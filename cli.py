#!/usr/bin/env python3
"""Factor research pipeline.

    python cli.py calibrate    does the pipeline report zero on pure noise?
    python cli.py audit        look-ahead check on every registered factor
    python cli.py evaluate     IC, cost curve, and walk-forward for each factor

Real China A-share data (needs `pip install -r requirements-data.txt`):

    python cli.py fetch --universe all          download and cache daily bars
    python cli.py --data data/ashare-all audit
    python cli.py --data data/ashare-all --horizon 5 evaluate
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alpha import evaluate as ev
from alpha import factors
from alpha.lookahead import detect
from alpha.panel import synthetic_panel


def _prepared(panel, name: str, real=None) -> pd.DataFrame:
    raw = factors.REGISTRY[name](panel)
    if real is None:
        # Neutralise against trailing volatility: a size/risk proxy that most
        # naive factors are accidentally loaded on.
        exposure = panel.close.pct_change().rolling(60).std()
        return factors.prepare(raw, exposure=exposure)
    # On A-shares size is the dominant style factor, so neutralise against it
    # directly. Mask before preparing, so each cross-section is standardised
    # over the stocks that could actually be traded that day.
    return factors.prepare(real.mask(raw), exposure=np.log(real.float_cap))


def _load(args):
    """(panel, AshareData or None, description) for the chosen data."""
    if args.data is None:
        panel = synthetic_panel(args.instruments, args.periods, seed=args.seed,
                                embed_signal=args.signal)
        return panel, None, "synthetic panel"
    from alpha import ashare

    real, manifest = ashare.load(Path(args.data), start=args.start, end=args.end)
    panel = real.panel
    lines = [
        f"A-share data from {args.data}: {manifest['universe']} universe, "
        f"{panel.close.shape[1]} stocks, {panel.timestamps[0]:%Y-%m-%d} to "
        f"{panel.timestamps[-1]:%Y-%m-%d} ({len(panel.timestamps)} sessions)",
        f"back-adjusted (hfq); {real.tradable.to_numpy().sum() / panel.close.notna().to_numpy().sum():.1%} "
        "of stock-days tradable after masking limits and new listings",
    ]
    if manifest["universe"] != "all":
        lines.append("WARNING: index universe uses TODAY's constituents -- survivorship "
                     "and selection bias; use --universe all for results you quote")
    return panel, real, "\n".join(lines)


def cmd_calibrate(args: argparse.Namespace) -> int:
    """The most important command here.

    An evaluation pipeline that reports a healthy IC on data with no structure
    is broken, and you cannot tell from looking at a number on real data. So
    run it on noise first and confirm it says nothing, then on data with a known
    planted signal and confirm it finds roughly what was planted.
    """
    print("A. pure noise -- a working pipeline must report ~0 IC\n")
    noise = synthetic_panel(args.instruments, args.periods, seed=args.seed, embed_signal=0.0)
    forward = noise.forward_return(1)
    for name in ("reversal_1", "momentum_20", "volatility_20"):
        summary = ev.summarize_ic(ev.information_coefficient(_prepared(noise, name), forward))
        flag = "ok" if abs(summary.mean) < 0.03 else "SUSPICIOUS"
        print(f"  {name:<20} IC {summary.mean:+.4f}  hit rate {summary.hit_rate:.3f}   {flag}")

    print(f"\nB. planted reversal of {args.signal} -- it must be recovered\n")
    planted = synthetic_panel(args.instruments, args.periods, seed=args.seed, embed_signal=args.signal)
    forward = planted.forward_return(1)
    for name in ("reversal_1", "momentum_20"):
        summary = ev.summarize_ic(ev.information_coefficient(_prepared(planted, name), forward))
        print(f"  {name:<20} IC {summary.mean:+.4f}  hit rate {summary.hit_rate:.3f}")

    print(
        "\nThe planted-signal numbers are a property of synthetic data with a\n"
        "reversal written into it by hand. They are a check that the machinery\n"
        "works, and say nothing whatsoever about any real market."
    )
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from alpha import ashare

    cache = Path(args.cache or f"data/ashare-{args.universe}")
    listing = cache / "universe.csv"
    if listing.exists():  # freeze the universe on the first run, so resumes agree
        codes = pd.read_csv(listing, dtype={"code": str})
    else:
        for attempt in range(5):
            try:
                codes = ashare.universe(args.universe)
                break
            except Exception as error:  # listing endpoints drop connections too
                print(f"  listing failed ({type(error).__name__}), retrying")
                time.sleep(5 * 2 ** attempt)
        else:
            raise SystemExit("could not fetch the universe listing; try again later")
        cache.mkdir(parents=True, exist_ok=True)
        codes.to_csv(listing, index=False)
    print(f"{len(codes)} codes in the {args.universe} universe -> {cache}/")
    failed = ashare.fetch(codes["code"], cache, start=args.start, end=args.end, pause=args.pause)
    ashare.write_manifest(cache, universe=args.universe, start=args.start, end=args.end,
                          adjust="hfq", source="akshare: Sina stock_zh_a_daily, Eastmoney stock_zh_a_hist fallback",
                          codes=int(len(codes)), sources=codes["source"].value_counts().to_dict(),
                          failed=failed)
    print(f"done; {len(failed)} codes failed (listed in manifest.json; re-run to retry)")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    panel, real, description = _load(args)
    if real is not None:
        print(description + "\n")
    print(f"{'factor':<24}{'look-ahead':<14}{'max drift'}")
    for name, fn in factors.REGISTRY.items():
        report = detect(fn, panel, cut_points=8, warmup=80)
        print(f"{name:<24}{'clean' if report.clean else 'LEAKS':<14}{report.max_drift:.3e}")

    leaky = lambda p: -p.close.pct_change(1).shift(-1)  # noqa: E731
    report = detect(leaky, panel, cut_points=8, warmup=80)
    print(f"\n{'(deliberate leak)':<24}{'clean' if report.clean else 'LEAKS':<14}{report.max_drift}")
    print("The last row is a factor written to peek one day ahead. If the audit\n"
          "cannot flag that, the audit is not doing anything.")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    panel, real, description = _load(args)
    h = args.horizon
    if real is not None or h != 1:  # keep the synthetic default output unchanged
        print(description)
    per_year = ev.TRADING_DAYS / h
    # Rebalance every h sessions: sample one row in h so periods do not overlap.
    forward = panel.forward_return(h).iloc[::h]
    if real is not None or h != 1:
        print(f"horizon {h} session(s): rebalanced every {h}, {len(forward)} periods\n")

    candidates = {name: _prepared(panel, name, real).iloc[::h] for name in factors.REGISTRY}
    if real is not None:
        # Small-cap tilt, the best-known A-share anomaly. Not neutralised: it
        # *is* the exposure everything else is neutralised against.
        candidates["size (small cap)"] = factors.prepare(
            real.mask(-np.log(real.float_cap))).iloc[::h]

    print(f"{'factor':<24}{'IC':>9}{'ICIR':>9}{'hit':>8}{'decay':>9}"
          f"{'turnover':>10}{'Sharpe@0':>10}{'Sharpe@10bp':>13}{'breakeven':>11}")
    for name, prepared in candidates.items():
        ic = ev.information_coefficient(prepared, forward)
        if ic.dropna().empty:
            continue
        summary = ev.summarize_ic(ic, prepared, periods_per_year=per_year)
        weights = ev.quantile_weights(prepared, args.quantiles)
        curve = ev.cost_curve(weights, forward, periods_per_year=per_year)
        free = curve.loc[curve["cost_bps"] == 0.0, "sharpe"].iloc[0]
        realistic = curve.loc[curve["cost_bps"] == 10.0, "sharpe"].iloc[0]
        breakeven = ev.breakeven_cost_bps(curve)
        print(f"{name:<24}{summary.mean:>9.4f}{summary.icir:>9.2f}{summary.hit_rate:>8.3f}"
              f"{summary.autocorr_1:>9.3f}{curve['turnover'].iloc[0]:>10.3f}"
              f"{free:>10.2f}{realistic:>13.2f}{breakeven:>11.1f}")

    print("\nwalk-forward, best factor by in-sample IC:")
    best = max(
        candidates,
        key=lambda n: abs(ev.information_coefficient(candidates[n], forward).mean()),
    )
    table = ev.walk_forward(candidates[best], forward,
                            train=max(250 // h, 20), test=max(60 // h, 4))
    if not table.empty:
        print(f"  {best}: in-sample IC {table['ic_in_sample'].mean():+.4f}, "
              f"out-of-sample {table['ic_out_of_sample'].mean():+.4f} "
              f"over {len(table)} windows")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instruments", type=int, default=60)
    parser.add_argument("--periods", type=int, default=750)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--signal", type=float, default=0.15,
                        help="strength of the reversal planted in synthetic data")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--data", help="cached A-share directory from `fetch`; omit for synthetic")
    parser.add_argument("--start", help="first date to use from --data, YYYY-MM-DD")
    parser.add_argument("--end", help="last date to use from --data, YYYY-MM-DD")
    parser.add_argument("--horizon", type=int, default=1,
                        help="holding period in sessions; 5 = weekly rebalancing")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate", help="noise gives nothing, planted signal is recovered").set_defaults(func=cmd_calibrate)
    sub.add_parser("audit", help="look-ahead check on every factor").set_defaults(func=cmd_audit)
    sub.add_parser("evaluate", help="IC, cost curve, walk-forward").set_defaults(func=cmd_evaluate)
    fetch = sub.add_parser("fetch", help="download A-share daily bars via AkShare")
    fetch.add_argument("--universe", choices=["csi300", "csi500", "all"], default="all")
    fetch.add_argument("--start", default="2016-01-01")
    fetch.add_argument("--end", default="2025-12-31")
    fetch.add_argument("--cache", help="directory (default data/ashare-<universe>)")
    fetch.add_argument("--pause", type=float, default=0.6, help="seconds between requests")
    fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
