# audit, A-shares, csi300 universe (today's constituents: biased, see README)

    command: python cli.py --data data/ashare-csi300 audit
    python:  Python 3.12.14
    numpy:   2.5.3
    pandas:  3.0.6
    akshare: 1.19.1
    data:    akshare: Sina stock_zh_a_daily, Eastmoney stock_zh_a_hist fallback
    fetched: 2026-10-01T00:58:50+00:00, 300 codes requested, 0 not obtainable
    date:    2026-10-01T01:01:40Z

```
A-share data from data/ashare-csi300: csi300 universe, 300 stocks, 2016-01-04 to 2025-12-31 (2430 sessions)
back-adjusted (hfq); 96.4% of stock-days tradable after masking limits and new listings
WARNING: index universe uses TODAY's constituents -- survivorship and selection bias; use --universe all for results you quote

factor                  look-ahead    max drift
reversal_1              clean         0.000e+00
reversal_5              clean         0.000e+00
momentum_20             clean         0.000e+00
momentum_60_skip_5      clean         0.000e+00
volatility_20           clean         0.000e+00
volume_trend            clean         0.000e+00
price_volume_corr       clean         0.000e+00

(deliberate leak)       LEAKS         inf
The last row is a factor written to peek one day ahead. If the audit
cannot flag that, the audit is not doing anything.
```
