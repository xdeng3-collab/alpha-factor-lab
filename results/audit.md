# audit

    command: python cli.py audit
    python:  Python 3.9.6
    numpy:   2.0.2
    pandas:  2.3.3
    seed:    default (see cli.py --seed)
    date:    2026-09-15T03:53:29Z

```
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
