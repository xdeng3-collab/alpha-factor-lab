# evaluate

    command: python cli.py evaluate
    python:  Python 3.9.6
    numpy:   2.0.2
    pandas:  2.3.3
    seed:    default (see cli.py --seed)
    date:    2026-09-15T03:53:31Z

```
factor                         IC     ICIR     hit    decay  turnover  Sharpe@0  Sharpe@10bp  breakeven
reversal_1                 0.1351    16.58   0.851   -0.143     1.572     14.51         8.11       22.2
reversal_5                 0.0652     8.21   0.718    0.741     0.750      6.65         3.28       19.7
momentum_20               -0.0316    -3.99   0.390    0.932     0.389     -3.60        -5.32        0.0
momentum_60_skip_5        -0.0006    -0.08   0.509    0.974     0.249     -0.21        -1.35        0.0
volatility_20             -0.0006    -0.07   0.479    0.937     0.331     -0.15        -1.61        0.0
volume_trend               0.0058     0.71   0.527    0.794     0.627      0.82        -1.99        2.9
price_volume_corr         -0.0005    -0.06   0.498    0.949     0.309     -0.12        -1.58        0.0

walk-forward, best factor by in-sample IC:
  reversal_1: in-sample IC +0.1364, out-of-sample +0.1383 over 8 windows
```
