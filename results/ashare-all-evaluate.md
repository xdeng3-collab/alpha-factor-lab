# evaluate, A-shares, all universe

    command: python cli.py --data data/ashare-all --horizon 5 evaluate
    python:  Python 3.12.14
    numpy:   2.5.3
    pandas:  3.0.6
    akshare: 1.19.1
    data:    akshare: Sina stock_zh_a_daily, Eastmoney stock_zh_a_hist fallback
    fetched: 2026-10-04T06:17:00+00:00, 5933 codes requested, 709 not obtainable
    date:    2026-10-04T06:22:26Z

```
A-share data from data/ashare-all: all universe, 5163 stocks, 2016-01-04 to 2025-12-31 (2430 sessions)
back-adjusted (hfq); 95.3% of stock-days tradable after masking limits and new listings
WARNING: 361 of 361 delisted stocks could not be fetched -- survivorship bias remains in this universe
horizon 5 session(s): rebalanced every 5, 486 periods

factor                         IC     ICIR     hit    decay  turnover  Sharpe@0  Sharpe@10bp  breakeven
reversal_1                 0.0148     0.95   0.571   -0.011     1.520      0.73        -0.67        5.2
reversal_5                 0.0345     2.19   0.630   -0.046     1.531      1.33        -0.03        9.8
momentum_20               -0.0538    -2.93   0.336    0.696     0.819     -1.62        -2.24        0.0
momentum_60_skip_5        -0.0341    -1.80   0.400    0.876     0.538     -0.87        -1.25        0.0
volatility_20              0.0747     3.37   0.708    0.894     0.470      1.25         0.95       41.6
volume_trend              -0.0497    -3.26   0.323    0.658     0.894     -1.44        -2.23        0.0
price_volume_corr          0.0584     5.01   0.780    0.759     0.729      2.28         1.40       25.9
size (small cap)           0.0138     0.53   0.548    0.998     0.088      0.54         0.49        inf

walk-forward, best factor by in-sample IC:
  volatility_20: in-sample IC +0.0725, out-of-sample +0.0727 over 36 windows
```
