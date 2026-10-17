# Architecture

Autonomous bot for Binance USD-M futures (BTCUSDT, ETHUSDT). Gradient-boosting models predict direction and expected move on the 4h candle close. A hard-coded risk layer sizes, vetoes or shrinks every signal before the order manager sends it.

## Data flow

```
Binance REST / data.binance.vision / WebSocket
        │  (klines 1h/4h/1d, funding, OI + long/short metrics, liquidations)
        ▼
 TimescaleDB ──► features (as-of joins, no look-ahead) ──► Ensemble (CatBoost + LightGBM)
                                                             │ p_short/p_flat/p_long, pred_ret
                                                             ▼
                                         make_signal (confidence ≥ threshold, regressor agrees)
                                                             ▼
                                         plan_trade (risk engine: size, ATR stop, TP, leverage cap)
                                                             ▼
                                         OrderManager → Binance (entry + stop + take-profit)
```

The same feature builder and risk engine are used for training, backtesting and live trading.

## Module map (`src/trader_bot/`)

| Package | Responsibility |
|---|---|
| `config/` | `Settings` (.env) and `StrategyConfig` (`config/strategy.yaml`); risk values are clamped to hard limits on load |
| `db/` | SQLAlchemy Core schema, Timescale hypertables, idempotent `upsert`, `load_frame` |
| `data/sources/` | `binance_rest` (public market data), `binance_archive` (historic metrics), `adapters` (on-chain/liquidation provider stubs) |
| `data/ingest/` | `backfill` (idempotent history + gap detection), `live` (WebSocket collectors + polling) |
| `features/` | `builder` (leakage-safe features), `pipeline` (DB → features) |
| `labeling/` | Forward-return labels and regression target in `labels.py`; triple-barrier labels in `barriers.py` |
| `models/` | `ensemble`, `barrier` (calibrated binary models), `barrier_experiment`, `cv` (purged walk-forward), `train` (CV, Optuna, walk-forward predictions), `registry`, `retrain` |
| `backtest/` | `engine` (bar simulator), `metrics`, `run` (classic walk-forward report), `segments`, `run_barrier`, `baselines` |
| `risk/` | `limits` (hard maxima), `engine` (signal + sizing), `kill_switch` |
| `execution/` | `client` (Binance SDK adapter), `filters`, `manager` (orders, reconcile, exits), `user_stream` |
| `scheduler/` | `runner`: 4h live loop and weekly retrain |
| `cli.py` | Typer commands |

## Database tables

Hypertables: `klines`, `funding`, `oi_metrics`, `liquidations`, `onchain`. Plain tables: `signals` (every decision, accepted or not, with reason) and `orders` (defined, not yet written to).

Schema is created by `trader-bot init-db` (`create_all` plus `create_hypertable`), not Alembic migrations.

## Runtime state (files)

| Path | Purpose |
|---|---|
| `state/KILL_SWITCH.lock` | Present = bot refuses to start |
| `state/anchors.json` | Day/week starting equity for loss limits |
| `state/positions.json` | Bot's open trades (for time exits across restarts) |
| `artifacts/models/<symbol>/<version>/` | `ensemble.joblib`, `meta.json`; `current.json` points to the live version |
| `artifacts/backtest/` | `report.json`, `equity.csv`, `trades.csv`, per-symbol predictions |

## Deployment

`docker-compose.yml`: `timescaledb` (always), `collector` (`trader-bot collect`), `bot` (profile `bot`, `restart: "no"` so a kill-switch halt is never auto-restarted).
