# alpha-factor-lab

Cross-sectional factor research where look-ahead bias is caught by the machinery
rather than by careful reading.

```bash
pip install -r requirements.txt
python -m pytest tests -q       # 37 tests, all offline
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

## Real data: China A-shares

```bash
pip install -r requirements-data.txt
python cli.py fetch --universe all                  # every listed + delisted A-share, cached
python cli.py --data data/ashare-all audit
python cli.py --data data/ashare-all --horizon 5 evaluate
```

`alpha/ashare.py` downloads back-adjusted daily bars through AkShare (Sina, with
Eastmoney as the fallback), caches one CSV per stock under `data/` (ignored by
git), and builds the same `Panel` the synthetic pipeline uses. The data layer,
not the factors, is where a real market's traps are handled:

* **Back-adjusted, never forward-adjusted.** Forward adjustment (前复权)
  rewrites every historical price each time a new dividend is paid, so a 2019
  price computed from a vendor in 2025 already contains 2025 information. It is
  look-ahead living in the data vendor, which no amount of careful factor code
  can see.
* **Untradable days are masked.** A stock that is suspended, or that closed at
  its daily price limit (10%; 20% for STAR and, from 24 Aug 2020, ChiNext; 30%
  on the Beijing exchange), could not have been bought or sold at that close.
  Its factor value is blanked for that day, *before* the cross-section is
  standardised, so the backtest never trades it.
* **No prices across gaps.** Suspended days are NaN, not forward-filled. A price
  carried across a suspension is a price nobody could trade at, and the return
  that spans the gap is dropped rather than credited.
* **New listings are excluded** for their first 60 sessions, which run without
  normal limits and carry returns no factor is meant to explain.
* **Size is the neutraliser.** Float market cap is recovered point-in-time from
  the same row as traded amount / turnover rate, and every factor is
  neutralised against its log. On A-shares the small-cap premium is large
  enough that an un-neutralised factor is often a size bet in disguise.
* **Weekly rebalancing.** `--horizon 5` holds for five sessions and samples one
  row in five so periods never overlap. Daily rebalancing at A-share costs
  (5 bps stamp duty on sales plus commission) is rarely what anyone would run.

### What it found, 2016–2025

Weekly rebalancing, every factor neutralised against size, 5,163 stocks
traded on Shanghai and Shenzhen. Full output with provenance in
[`results/ashare-all-evaluate.md`](results/ashare-all-evaluate.md).

```
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

* **Low volatility and price-volume divergence survive costs.** They are the
  only two with a positive Sharpe at 10 bps, and their breakeven costs (42 and
  26 bps) sit well above what an institution pays. Walk-forward IC for low
  volatility is the same in and out of sample, so it is not one lucky period.
* **Momentum is negative.** On A-shares, one- and three-month winners
  underperform. This matches the published literature on China's
  retail-dominated market rather than the US pattern the factor names come
  from. Read as a reversal signal it would look strong, but that is choosing
  the sign after seeing the answer, and it is reported here as it was defined.
* **Short-horizon reversal dies at realistic cost.** Weekly turnover of 1.5
  eats the edge by 5–10 bps.
* **Costs are a sweep, not a model.** Real A-share cost is roughly 5 bps
  stamp duty on sales plus commission and impact; read the 10 bps column as
  "cheap institutional" and the breakeven column as the margin.

The same run on today's CSI 300 constituents
([`results/ashare-csi300-evaluate.md`](results/ashare-csi300-evaluate.md)) gives
much weaker numbers throughout, which is the expected shape: large caps are
more efficiently priced, and a 300-stock cross-section is noisier.

What it does *not* handle, stated so nobody has to discover it:

* **Survivorship bias is not fixed, only measured.** `--universe all` asks for
  every delisted stock the exchanges list (361), but Sina serves none of them
  and Eastmoney refused connections from the network this ran on, so the
  results above contain **0 of 361** delisted stocks, and none of the 348
  Beijing listings. The CLI prints this on every run and `manifest.json`
  records each failed code and why. Delisted stocks skew towards distressed,
  volatile names, so this most likely bears on the volatility and reversal
  results; which way it moves them is not something this data can say. The `csi300`/`csi500` universes use *today's* constituents and are
  biased towards stocks that grew into the index; the CLI prints a warning
  whenever they are used.
* **ST stocks** trade with a 5% limit, and ST status history is not in this
  data, so an ST stock pinned at 5% is not masked.
* **Holding through a limit-down** is not modelled: a stock bought earlier and
  then locked limit-down is assumed sellable at the close.
* **Multiple testing.** Seven factors are evaluated here and all seven are
  reported. The number of things tried belongs next to any result quoted.

## Layout

```
alpha/
  panel.py       point-in-time panel; asof() is the only reader
  lookahead.py   truncate-and-compare detection
  factors.py     seven factors, winsorize / neutralize / standardize
  evaluate.py    IC, quantile backtest, cost curve, walk-forward
  ashare.py      A-share download, cache, tradability mask, size
cli.py           calibrate / audit / evaluate
tests/           37 tests, weighted toward the detector, the noise calibration and the A-share mask
```

## License

MIT
