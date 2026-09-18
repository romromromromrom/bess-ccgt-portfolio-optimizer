# Multi-Asset Energy & Ancillary Services Optimizer

[![CI/CD](https://github.com/romromromromrom/bess-ccgt-portfolio-optimizer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/romromromromrom/bess-ccgt-portfolio-optimizer/actions/workflows/ci-cd.yml)

I built this to answer a question I kept circling back to: how much is it actually
worth to a portfolio to co-optimize energy and reserve markets, instead of bidding
into each one separately? This is a MILP model — Pyomo + HiGHS, fully open-source,
no commercial solver licence needed — that answers it for a battery (BESS) and a
CCGT trading energy, FCR and aFRR at the same time.

> **Technical demonstrator, not a production bidding system.** The market data is
> synthetic. Read [Limitations](#limitations) before treating any number below as
> a forecast.

## What it does

For every hour of a price horizon, and for both assets at once, the model decides
how much energy to buy or sell and how much capacity to hold back and sell as
reserve — subject to the physical constraints of each asset. From that it:

- quantifies what co-optimizing energy and reserves is actually worth, compared
  to trading energy alone,
- turns the resulting schedule into a bid table an operator could plausibly submit,
- backtests the committed decision against price data the model never saw while
  deciding,
- and serves the whole thing through a Streamlit app, containerized and set up
  to deploy to AWS.

## The result worth explaining

On the default 7-day synthetic horizon, adding reserve markets to the picture
roughly doubles the portfolio's P&L:

| | Energy only | Energy + Reserves |
|---|---:|---:|
| Portfolio P&L | 359,309 EUR | **693,193 EUR** |
| of which reserve revenue | 0 | 507,855 EUR |
| Uplift | — | **+333,884 EUR (+93%)** |

The number is less interesting than the behaviour behind it: the CCGT commits for
106 hours over the week, and for 71 of those hours it runs with the spot price
*below* its own short-run marginal cost — losing an average of 2,254 EUR/h on the
energy leg. It does this on purpose, because sitting near Pmin frees up the
Pmax−Pmin headroom to sell as upward aFRR capacity, and that revenue more than
covers the loss.

I didn't hard-code that behaviour anywhere — it falls out of the objective once
energy and reserve are solved jointly. An energy-only model, or two models solved
market by market, has no way to see this trade-off at all.

![Energy only vs Energy + Reserves](docs/screenshot-comparison.png)

![Dispatch](docs/screenshot-dispatch.png)

## Markets and assets

| | Products | Remuneration |
|---|---|---|
| **Energy** | day-ahead spot | EUR/MWh delivered |
| **FCR** | symmetric | capacity only (EUR/MW/h) |
| **aFRR** | up, down | capacity (EUR/MW/h) **+** activation (EUR/MWh, uncertain) |

| Asset | Modelled as |
|---|---|
| **BESS** | power/energy limits, split charge & discharge efficiency, SOC dynamics, cycling degradation cost, mode-exclusivity binary |
| **CCGT** | unit commitment (on/start/stop binaries), Pmin/Pmax, heat rate, CO2, VOM, startup cost, ramp limits, min up/down time |

## Mathematical formulation

Both assets are MILPs. Two constraint families are what actually make this a
co-optimization rather than two separate problems bolted together:

**Power headroom** — a MW sold as reserve is a MW that's no longer available for
arbitrage.

```
BESS :  p_net + Σ reserve_up   ≤  max_discharge
        p_net − Σ reserve_down ≥ −max_charge          where p_net = discharge − charge

CCGT :  p + Σ reserve_up   ≤  Pmax · on
        p − Σ reserve_down ≥  Pmin · on
```

**Energy headroom (BESS only)** — a reserved MW has to be *sustainable*, not just
physically available for an instant. The battery needs enough charge behind it to
deliver upward reserve for `sustain_duration_h`, and enough empty room to absorb
downward reserve — checked at both the opening and closing SOC of every period:

```
Σ reserve_up   · sustain ≤ soc − soc_min
Σ reserve_down · sustain ≤ soc_max − soc
```

**Objective** — maximise energy margin minus costs plus reserve value, where
holding 1 MW of reserve for 1 hour is worth:

```
capacity_price  +  ratio × (activation_price − opportunity_cost)      [up]
capacity_price  +  ratio × (opportunity_cost − activation_price)      [down]
capacity_price                                                        [symmetric]
```

The part I'd flag myself if someone asked about this model in an interview:
activation energy is *not* free revenue. Getting called upward means the CCGT
burns gas it wouldn't otherwise burn, and the battery gives up a spot sale it
could have made instead — so activation is valued at its margin over the
opportunity cost (SRMC for the CCGT, spot price + degradation for the battery),
never at the raw activation price. A symmetric product like FCR ends up
energy-neutral in expectation, which also matches why FCR in France is
remunerated on capacity alone.

That valuation logic lives in exactly one place (`valuation.py`), used by the
MILP objective, the P&L attribution and the backtest — so the optimizer's
"expected P&L" and the backtest's "expected" leg are guaranteed to be the same
number, not just approximately close. A test pins that agreement to 1e-9.

## Backtesting

Two phases, kept strictly separate:

1. **Optimize** on forecast prices and assumed activation ratios → this produces
   a *decision*.
2. **Re-value** that same decision, unchanged, against realized prices, realized
   fuel costs and the activation that was actually called.

The schedule is never re-solved with hindsight. Re-optimizing after the fact is
probably the single most common way a backtest ends up manufacturing skill a
strategy never actually had.

```
forecast_error_cost = expected_pnl − realized_pnl
```

On the default horizon this splits in a way I found genuinely informative: the
BESS loses about 11 kEUR to forecast error, the CCGT loses 191 kEUR. That's not
a coincidence — the battery's revenue is mostly reserve-capacity revenue, which
barely depends on the spot price, while the CCGT's revenue is dominated by the
energy margin, which is very sensitive to it. The asymmetry isn't something I
assumed going in; the model produced it.

![Expected vs realized](docs/screenshot-backtest.png)

## Automatic bidding

`bids.py` turns a solved schedule into one row per timestamp / asset / market /
direction / volume / price. The CCGT bids its own short-run marginal cost into
the energy market — bidding the spot price back at the market wouldn't do
anything useful. Reserve is bid at the product's capacity price. Bids under a
minimum volume are dropped, mostly because MILP solutions leave behind numerical
residue (a "bid" of 1e-9 MW isn't a bid), and real markets have a minimum
increment anyway.

## Architecture

```
src/energy_portfolio/
├── config.py          CCGTConfig, ReserveProductConfig  (capacity vs activation, kept separate)
├── bess_config.py     BESSReserveConfig
├── valuation.py       what 1 MW of reserve is worth — shared by MILP, P&L and backtest
├── bess_reserve.py    MILP: arbitrage + FCR/aFRR with power & energy headroom
├── ccgt.py            MILP: unit commitment + reserve headroom
├── portfolio.py       solve both, attribute P&L, compare energy-only vs co-optimized
├── backtest.py        expected vs realized on a fixed decision
├── bids.py            schedule -> bid table
├── market_data.py     SYNTHETIC data generator (seeded, replaceable)
├── s3_store.py        DataFrame <-> S3
└── solver.py          HiGHS via Pyomo APPSI
app/
├── streamlit_app.py   UI
└── charts.py          Altair builders
```

## Deployment

Runs live on Streamlit Community Cloud from `app/streamlit_app.py`. That platform
installs from `requirements.txt` but doesn't actually build the project as a
package, so the app adds `src/` to `sys.path` at import time and runs straight
from a bare checkout. `pyproject.toml` stays the source of truth for local
development, Docker and CI — the sys.path trick exists only for the one platform
that needs it.

## How to run locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                             # 45 tests
streamlit run app/streamlit_app.py    # http://localhost:8501
```

## Tests

45 tests, all green. The ones I'd point to if asked which actually matter:

- `test_starts_at_pmin_below_marginal_cost_to_sell_reserve` — pins the headline
  economic behaviour, and checks that the energy-only run stays off at the same
  prices.
- `test_expected_leg_reproduces_the_optimizer_pnl` — optimizer and backtest agree
  to 1e-9.
- `test_perfect_foresight_leaves_no_forecast_error` — realized == forecast implies
  zero error cost.
- `test_access_to_reserve_markets_can_never_destroy_value` — adding a market can
  only add options; if this ever failed, the formulation would be wrong somewhere.
- `test_reserve_never_exceeds_energy_headroom` / `..._power_headroom` — the
  coupling constraints actually hold.
- `test_backtest_does_not_re_optimize_the_schedule` — the committed decision
  survives untouched.
- `test_streamlit_app_runs_without_exception` — boots the real app under
  `AppTest` and walks every tab.

```bash
pytest -q                    # everything
pytest -q -m "not slow"      # skip the Streamlit boot test
```

## Docker

```bash
docker compose up -d --build     # http://localhost:8501
```

## Cloud architecture

```
GitHub  ──push main──▶  GitHub Actions  ──pytest──▶  ssh + docker compose  ──▶  EC2 (t3.micro)
                                                                                 │
                                                          instance IAM role ─────┤
                                                                                 ▼
                                                                        S3 (market data, results)
```

Credentials are never read from the code or committed to the repo. boto3 picks
them up from the environment locally and from an EC2 instance role once
deployed, so there's no static access key sitting in a file on the server at
any point.

## Limitations

I'd rather list these out plainly than have someone find them the hard way —
knowing where a model stops matters more than the model itself:

- **The data is synthetic.** Seeded, shaped to look like a French market (double
  daily peak, midday solar dip, cheaper weekends, the occasional scarcity
  spike), but not sourced from RTE or ENTSO-E. Every headline number above
  inherits that. Drop in a real CSV with the same columns and nothing
  downstream needs to change.
- **BESS and CCGT are optimized independently**, then aggregated. There's no
  joint constraint coupling them — a shared grid connection limit, a single
  aggregated reserve obligation, a portfolio risk budget would each introduce
  one. That's the most obvious next step.
- **Reserve capacity prices are a horizon average per product**, not an hourly
  curve. Real auctions clear per block, so this is a first approximation.
- **Activation is an assumed expected ratio**, re-evaluated against a drawn
  realized ratio in the backtest. It doesn't feed back into SOC or fuel burn
  within the optimization period itself.
- **One sustain duration per product** stands in for the actual
  per-product/per-TSO prequalification rules.
- **No detailed CCGT startup trajectory** — ramp limits are relaxed on
  start/stop periods, so the unit reaches Pmin instantly, which is optimistic.
- **Bid prices are marginal-cost based**, with no opportunity-cost adder from
  the SOC shadow price or a water value.
- **Volume caps are a blunt proxy** for market depth, not a real
  bid-into-a-supply-curve formulation.

## Next steps

Roughly in the order I'd tackle them:

1. Swap in real RTE / ENTSO-E data instead of the generator.
2. Add a joint portfolio-level reserve constraint coupling the two assets.
3. Hourly reserve capacity price curves in the MILP — the objective is already
   linear in reserve volume, so this should be a small change.
4. Pull the SOC shadow price out of the MILP duals and use it as an
   opportunity-cost adder on bids.
5. Add hydro / pumped storage / wind as further asset classes.
6. A small FastAPI service alongside the Streamlit UI, for anything that needs
   to call this programmatically.
