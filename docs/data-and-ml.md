# Data, features and ML

## Data sources

| Data | Source | History |
|---|---|---|
| Klines 1h/4h/1d | Binance REST `/fapi/v1/klines` (closed candles only) | Full; backfilled from `history_start` |
| Funding rate | REST `/fapi/v1/fundingRate` | Full |
| Open interest, top/global long-short ratio, taker volume ratio (5m) | `data.binance.vision` daily archive; live via REST `/futures/data/*` | Archive for history (REST keeps only ~30 days) |
| Liquidations | Binance WebSocket `!forceOrder@arr`; API BRICKS Metrics V1 for historical 4h aggregates | Live events include position side; historical aggregates are total-only and loaded by `backfill-liquidations` |
| On-chain (inflow/outflow, whales, NUPL, MVRV-Z) | `OnchainProvider` adapter (Glassnode/CryptoQuant) | Stub, disabled without a key |
| Coinglass liquidations | `LiquidationProvider` adapter | Stub, disabled without a key |

Backfill is idempotent (upserts) and resumes from the last stored timestamp. `report_gaps` lists missing candles/metric intervals.

Because of the liquidation and on-chain limits, those feature groups default to **off** in `config/strategy.yaml` (`features.groups`).

### API BRICKS historical liquidation data

The backfill downloads `LIQUIDATION_QUANTITY` and `LIQUIDATION_AVERAGE_PRICE` at a 4-hour interval for BTCUSDT and ETHUSDT from the CoinAPI/API BRICKS Metrics V1 API. The configured endpoint does not expose `LIQUIDATION_SIDE` for these Binance perpetual symbols, so historical rows are stored with `side=UNKNOWN`; long/short is never inferred from total aggregates. Binance live `forceOrder` messages continue to map `SELL` to a liquidated long and `BUY` to a liquidated short.

The USD value is an estimate: the API returns quantity and average price as separate aggregates, and the loader multiplies them. If a bucket has quantity but no usable price, its USD value stays missing while quantity remains available. Empty 4-hour buckets are stored as zeros only between the first and last periods returned by the API. Values outside that downloaded coverage remain missing. The feature builder exposes total USD and quantity over 4h and rolling 24h, separately from directional live-event features, and expires stale values after their source interval.

Use `API_BRIKS_API_KEY` from `.env` and run `uv run trader-bot backfill-liquidations --start 2022-06-01 --end YYYY-MM-DD`. Earlier samples estimate roughly 350 REST credits for both symbols and both metrics over the full range; actual billing depends on the account plan and returned rows. Check the provider dashboard after backfill. Prior sample responses showed coverage beginning around June 2022. The group remains disabled by default because the first same-data walk-forward comparison produced no trades and did not pass the backtest gate.

References: [API BRICKS pricing](https://www.coinapi.io/products/market-data-api/pricing), [metrics catalog](https://www.coinapi.io/products/market-data-api/docs/metadata-tables/metric_id), [API BRICKS liquidation metrics overview](https://www.coinapi.io/blog/crypto-liquidation-data), [Binance liquidation order stream payload](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Liquidation-Order-Streams).

## Features (`features/builder.py`)

Rows are indexed by **decision time** = close of a 4h candle. Each source is computed on its native grid with trailing-only operations, stamped with the time it becomes available, then attached with `merge_asof(direction="backward")`. That is also how slow series are forward-filled.

- **price:** log returns (1/3/6/12/42 candles), volatility, ATR%, range, close position, volume z-score, taker buy ratio; 1h features (returns, vol) and 1d features (returns, vol, distance to SMA20), only from candles already closed.
- **derivatives:** funding rate and rolling means; OI value and its 1h/4h/24h change; long-short and taker ratios.
- **liquidations (optional):** historical total estimated USD and quantity over 4h/24h; long/short USD over 1/4/24h and imbalance only where side-labelled Binance events exist.
- **onchain (optional):** raw value and 7-day change, shifted by `onchain_publication_lag_hours` before forward-fill.

Columns prefixed `ctx_` (`ctx_close`, `ctx_atr`) are context for labeling/risk and are never model inputs. A test asserts features at time t are unchanged if all later data is removed.

## Labels (`labeling/labels.py`)

For horizon `h` (default 6 candles = 24h):
- `fwd_ret = close[t+h] / close[t] - 1` (regression target)
- class: +1 if `fwd_ret > band`, -1 if `< -band`, else 0, with `band = flat_atr_mult * atr_pct * sqrt(h)`

The last `h` rows have no label and are never used for training.

## Models (`models/`)

- **Ensemble:** LightGBM + CatBoost, probabilities averaged (classifier: short/flat/long) and returns averaged (regressor).
- **Validation:** purged expanding-window walk-forward. Training stops `horizon + embargo` candles before each test block.
- **Tuning:** `train --tune` runs Optuna per model family, minimising purged-CV multi-class logloss.
- **Label/imbalance experiment:** `uv run trader-bot experiment` compares flat ATR multipliers 0.5, 1.0, and 1.5 with classifiers unweighted and balanced. It writes fold-level metrics to `artifacts/experiments/<run>/report.json`; this is CV analysis, not a promotion or trading recommendation.
- **Signal threshold study:** `uv run trader-bot signal-study --holdout-start 2025-10-01` reports actual/predicted class balance and probability distributions, searches confidence and minimum reward/risk grids on earlier validation predictions, then evaluates the selected pair on the final holdout. Validation labels whose horizon overlaps the holdout are purged. It does not modify strategy settings or promote models.
- **Registry:** each training run saves a versioned artifact. A new version is promoted only if its CV logloss is no worse than the current one (the comparison is approximate, as the evaluation windows differ).
- **Retraining:** the live loop retrains weekly; `train` does it on demand.
- Not implemented: probability calibration.

## Signal rule

Trade only if the top class is long/short, confidence ≥ `confidence_threshold` (default 0.75) and the regressor's sign agrees. A 0.75 threshold on three classes will fire rarely; tune it from the backtest.

## Backtest (`backtest/`)

`trader-bot backtest` retrains walk-forward (every `retrain_every_candles`) on purged past data only, so every prediction is out-of-sample, then simulates trading from `backtest.start`.

Backtest overrides can be evaluated without changing the strategy defaults, for example `uv run trader-bot backtest --flat-atr-mult 0.5 --min-reward-risk 1.0 --balance-classes --out artifacts/backtest_candidate`. Reports record the settings used.

- Decision at candle close executes at the next candle's open.
- Within a bar, the stop is checked before the take-profit; a gap past the stop exits at the open.
- Costs: taker fee on entry and exit, slippage in bps, funding payments.
- Exit on stop, take-profit, or after `horizon` candles.
- Daily/weekly loss limits halt new entries.
- Capital is split equally across symbols (independent books); the portfolio curve is their sum.
- Metrics: Sharpe (daily returns, annualised √365), max drawdown, return, CAGR, win rate, profit factor.
- Reports include per-class out-of-sample counts/precision/recall and per-symbol backtest counts for predictions, signal filters, risk-plan rejections, and approved trades. `hc_coverage` includes confident flat predictions; use `directional_hc_coverage` and `directional_hc_precision` for confident directional predictions.
- **Gate:** Sharpe > 1.5 and |max drawdown| < 15% (`gate_passed` in `report.json`). Treat it as a check, not a target to optimise: keep a final untouched holdout period.
