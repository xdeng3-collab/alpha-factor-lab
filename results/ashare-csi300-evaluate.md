# evaluate, A-shares, csi300 universe (today's constituents: biased, see README)

    command: python cli.py --data data/ashare-csi300 --horizon 5 evaluate
    python:  Python 3.12.14
    numpy:   2.5.3
    pandas:  3.0.6
    akshare: 1.19.1
    data:    akshare: Sina stock_zh_a_daily, Eastmoney stock_zh_a_hist fallback
    fetched: 2026-10-01T00:58:50+00:00, 300 codes requested, 0 not obtainable
    date:    2026-10-01T01:01:45Z

```
A-share data from data/ashare-csi300: csi300 universe, 300 stocks, 2016-01-04 to 2025-12-31 (2430 sessions)
back-adjusted (hfq); 96.4% of stock-days tradable after masking limits and new listings
WARNING: index universe uses TODAY's constituents -- survivorship and selection bias; use --universe all for results you quote
horizon 5 session(s): rebalanced every 5, 486 periods

factor                         IC     ICIR     hit    decay  turnover  Sharpe@0  Sharpe@10bp  breakeven
reversal_1                 0.0090     0.41   0.505   -0.014     1.520      0.40        -0.56        4.1
reversal_5                 0.0134     0.57   0.535   -0.014     1.520      0.06        -0.87        0.7
momentum_20               -0.0169    -0.66   0.471    0.726     0.788      0.14        -0.29        3.2
momentum_60_skip_5        -0.0123    -0.44   0.478    0.896     0.497      0.11        -0.12        4.8
volatility_20              0.0215     0.71   0.541    0.913     0.428     -0.52        -0.72        0.0
volume_trend              -0.0113    -0.52   0.461    0.612     0.957      0.38        -0.23        6.2
price_volume_corr          0.0307     1.59   0.592    0.756     0.738      0.22        -0.32        4.1
size (small cap)           0.0002     0.01   0.493    0.998     0.083      0.88         0.84        inf

walk-forward, best factor by in-sample IC:
  price_volume_corr: in-sample IC +0.0314, out-of-sample +0.0297 over 36 windows
```
