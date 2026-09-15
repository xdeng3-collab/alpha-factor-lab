# calibrate

    command: python cli.py calibrate
    python:  Python 3.9.6
    numpy:   2.0.2
    pandas:  2.3.3
    seed:    default (see cli.py --seed)
    date:    2026-09-15T03:53:24Z

```
A. pure noise -- a working pipeline must report ~0 IC

  reversal_1           IC -0.0020  hit rate 0.492   ok
  momentum_20          IC -0.0008  hit rate 0.496   ok
  volatility_20        IC -0.0004  hit rate 0.508   ok

B. planted reversal of 0.15 -- it must be recovered

  reversal_1           IC +0.1351  hit rate 0.851
  momentum_20          IC -0.0316  hit rate 0.390

The planted-signal numbers are a property of synthetic data with a
reversal written into it by hand. They are a check that the machinery
works, and say nothing whatsoever about any real market.
```

Deterministic: same seed, same numbers on any machine. Nothing here is timed,
so unlike a latency benchmark this output is expected to reproduce exactly.
