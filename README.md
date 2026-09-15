# alpha-factor-lab

Cross-sectional factor research where look-ahead bias is caught by the machinery
rather than by careful reading.

```bash
pip install -r requirements.txt
python -m pytest tests -q       # 26 tests
python cli.py calibrate         # does the pipeline report nothing on noise?
python cli.py audit             # look-ahead check on every factor
python cli.py evaluate          # IC, cost curve, walk-forward
```


Every table in this README is the literal output of the command above it, and
the committed copies in [`results/`](results/) carry the interpreter and library
versions that produced them. Nothing here is timed, so unlike a latency
benchmark these numbers are expected to reproduce exactly on any machine — and
CI asserts they still do rather than merely checking the commands exit zero.

## The two things this is built around

**1. Look-ahead is structural, then proven.**

Every look-ahead bug is the same bug — something unknowable at time *t* ended up
in the factor value at *t* — and it never looks like a bug. It hides inside a
`shift` with the wrong sign or a `resample` with the wrong label, and code
review does not find it.

So `Panel.asof(t)` is the only way to read data and it cannot return a row after
*t*. A factor written against that interface is *incapable* of seeing ahead.

For factors not written that way, `alpha.lookahead` proves it empirically on an
absolute property: **if a factor's value at *t* used only information available
at *t*, then deleting every row after *t* cannot change it.** Recompute on a
truncated panel, compare the overlap. No tolerance band, no judgement call.

```
factor                  look-ahead    max drift
reversal_1              clean         0.000e+00
momentum_60_skip_5      clean         0.000e+00
price_volume_corr       clean         0.000e+00
...
(deliberate leak)       LEAKS         inf
```

That last row is a factor written to peek one day ahead, run on every audit. A
detector that only ever says "clean" would pass every test you could write for
it, so it has to be shown failing.

It catches the subtle cases too, not just a negative `shift`. Normalising by a
full-sample mean — `close / close.mean()` — is look-ahead, because every
historical value moves when tomorrow arrives. `tests/test_lookahead.py` includes
that one.

**2. Calibrate on noise before believing anything.**

`python cli.py calibrate` runs the whole pipeline on synthetic prices with *no
predictable structure*, then on prices with a reversal planted by hand:

```
A. pure noise -- a working pipeline must report ~0 IC
  reversal_1           IC -0.0020  hit rate 0.492   ok
  momentum_20          IC -0.0008  hit rate 0.496   ok
  volatility_20        IC -0.0004  hit rate 0.508   ok

B. planted reversal of 0.15 -- it must be recovered
  reversal_1           IC +0.1351  hit rate 0.851
  momentum_20          IC -0.0316  hit rate 0.390
```

A pipeline that finds signal in noise will find signal anywhere, and you cannot
tell from looking at a number on real data. Both halves are also pytest cases,
so the property is enforced rather than eyeballed.

**The planted-signal figures are a property of data with a reversal written into
it by hand. They say nothing about any real market, and this README is not
reporting an alpha.**

## What gets measured

* **IC** — Spearman rank correlation per cross-section, because the claim is
  about ordering, not linearity, and ranks are unbothered by the fat tails
  returns always have. Reported with ICIR, hit rate, and the factor's own
  one-period autocorrelation, which is how fast the signal decays.
* **Turnover and cost sensitivity** — Sharpe across 0 to 50 bps, plus the
  interpolated **breakeven cost**. A long-short book can have a perfectly good
  IC and still be entirely consumed by its own turnover, and a result that does
  not say where the edge dies is not a result.
* **Walk-forward** — in-sample against out-of-sample IC over rolling windows.
  Nothing is fitted, so the gap isolates one thing: how much of an IC was the
  period rather than the factor.

The pipeline is winsorize → neutralize → standardize, in that order. Clipping
first stops one outlier from dragging the neutralisation regression;
standardising last puts every factor on a common scale. Neutralisation usually
*lowers* the IC, and that drop is the information: it is the part of the signal
that was a risk-factor bet rather than stock selection.

Cost accounting charges the initial build. A backtest that starts fully invested
for free has already stolen its first period of cost; `test_putting_the_book_on_is_charged_for`
holds that line.

## Using real data

`Panel` takes any wide close/volume frames, so pointing this at real prices is a
loader away. Two things to write in your own README when you do:

* **Survivorship bias.** The convenient free sources only carry instruments that
  still exist. Every backtest on them is run on a universe selected for having
  survived. If you cannot fix it, say so.
* **Multiple testing.** Trying thirty factors and reporting the best three makes
  those three t-statistics meaningless. The number of things tried belongs next
  to the result.

Neither is handled here, because neither can be handled by a library — they are
claims about how the research was conducted.

## Layout

```
alpha/
  panel.py       point-in-time panel; asof() is the only reader
  lookahead.py   truncate-and-compare detection
  factors.py     seven factors, winsorize / neutralize / standardize
  evaluate.py    IC, quantile backtest, cost curve, walk-forward
cli.py           calibrate / audit / evaluate
tests/           26 tests, weighted toward the detector and the noise calibration
```

## License

MIT
