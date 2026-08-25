#!/usr/bin/env python3
"""Factor research pipeline.

    python cli.py calibrate    does the pipeline report zero on pure noise?
    python cli.py audit        look-ahead check on every registered factor
    python cli.py evaluate     IC, cost curve, and walk-forward for each factor
"""

from __future__ import annotations

import argparse

import pandas as pd

from alpha import evaluate as ev
from alpha import factors
from alpha.lookahead import detect
from alpha.panel import synthetic_panel


def _prepared(panel, name: str) -> pd.DataFrame:
    raw = factors.REGISTRY[name](panel)
    # Neutralise against trailing volatility: a size/risk proxy that most naive
    # factors are accidentally loaded on.
    exposure = panel.close.pct_change().rolling(60).std()
    return factors.prepare(raw, exposure=exposure)


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


def cmd_audit(args: argparse.Namespace) -> int:
    panel = synthetic_panel(args.instruments, args.periods, seed=args.seed)
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
    panel = synthetic_panel(args.instruments, args.periods, seed=args.seed,
                            embed_signal=args.signal)
    forward = panel.forward_return(1)

    print(f"{'factor':<24}{'IC':>9}{'ICIR':>9}{'hit':>8}{'decay':>9}"
          f"{'turnover':>10}{'Sharpe@0':>10}{'Sharpe@10bp':>13}{'breakeven':>11}")
    for name in factors.REGISTRY:
        prepared = _prepared(panel, name)
        ic = ev.information_coefficient(prepared, forward)
        if ic.dropna().empty:
            continue
        summary = ev.summarize_ic(ic, prepared)
        weights = ev.quantile_weights(prepared, args.quantiles)
        curve = ev.cost_curve(weights, forward)
        free = curve.loc[curve["cost_bps"] == 0.0, "sharpe"].iloc[0]
        realistic = curve.loc[curve["cost_bps"] == 10.0, "sharpe"].iloc[0]
        breakeven = ev.breakeven_cost_bps(curve)
        print(f"{name:<24}{summary.mean:>9.4f}{summary.icir:>9.2f}{summary.hit_rate:>8.3f}"
              f"{summary.autocorr_1:>9.3f}{curve['turnover'].iloc[0]:>10.3f}"
              f"{free:>10.2f}{realistic:>13.2f}{breakeven:>11.1f}")

    print("\nwalk-forward, best factor by in-sample IC:")
    best = max(
        factors.REGISTRY,
        key=lambda n: abs(ev.information_coefficient(_prepared(panel, n), forward).mean()),
    )
    table = ev.walk_forward(_prepared(panel, best), forward)
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

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate", help="noise gives nothing, planted signal is recovered").set_defaults(func=cmd_calibrate)
    sub.add_parser("audit", help="look-ahead check on every factor").set_defaults(func=cmd_audit)
    sub.add_parser("evaluate", help="IC, cost curve, walk-forward").set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
