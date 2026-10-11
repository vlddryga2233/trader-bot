# Operations

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker.

```bash
cp .env.example .env              # fill testnet keys
docker compose up -d timescaledb
uv sync
uv run trader-bot init-db
```

If `uv sync` fails with connection resets to `pypi.org`, your network proxy is blocking it; use an allowed index or another network.

## Typical workflow

```bash
uv run trader-bot backfill        # history + gap report
uv run trader-bot train --tune    # optional Optuna; promotes a model if not worse
uv run trader-bot backtest        # artifacts/backtest/report.json
uv run trader-bot collect         # keep running (or: docker compose up -d collector)
uv run trader-bot run --testnet   # trading loop on Binance Futures testnet
```

## CLI

| Command | Purpose |
|---|---|
| `init-db` | Create tables and hypertables |
| `backfill` | Download history, print gap counts |
| `backfill-liquidations --start YYYY-MM-DD --end YYYY-MM-DD` | Load API BRICKS 4h total liquidation aggregates for configured symbols; requires `API_BRIKS_API_KEY` |
| `collect` | Live klines, liquidations, metrics polling |
| `train [--tune]` | Train/CV/promote models for all symbols |
| `train --no-promote` | Save model candidates while keeping the active model versions unchanged |
| `backtest [--out DIR]` | Walk-forward OOS backtest and gate result |
| `signal-study --holdout-start YYYY-MM-DD` | Report class/probability balance, tune confidence and reward/risk on validation, then score the later holdout |
| `run --testnet` | Trading loop (testnet only) |
| `killswitch-reset` | Clear the kill-switch lock manually |

## Configuration

`.env` (see `.env.example`): `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET`, `DATABASE_URL`, optional `API_BRIKS_API_KEY` / `GLASSNODE_API_KEY` / `CRYPTOQUANT_API_KEY` / `COINGLASS_API_KEY`, `STRATEGY_CONFIG`, `STATE_DIR`, `ARTIFACTS_DIR`.

`config/strategy.yaml`: symbols, `history_start`, feature groups, label horizon/band, model settings (confidence threshold, CV splits, embargo, retrain cadence, Optuna trials), risk (clamped to hard limits) and backtest costs/start date.

## Testnet runbook

1. Create Binance Futures testnet keys; put them in `.env` with `BINANCE_TESTNET=true`.
2. Complete the workflow above and confirm the backtest gate on a period the models never saw.
3. `docker compose --profile bot up bot` (no auto-restart).
4. Watch logs for `trade.opened`, `order.update`, `trade.closed_by_exchange`, `trade.time_exit`.
5. Compare the `signals` table and testnet fills against the backtest's expectations before considering anything further.
6. If the kill switch fires: inspect the logs and positions, rotate keys if needed, then `killswitch-reset`.

## Tests

```bash
uv run pytest
```

Covers leakage, purged CV, risk sizing and clamping, kill switch, backtest accounting, exchange filters and the order manager (with a fake exchange).

## Current status / known gaps

- Dependencies could not be installed in the build environment (PyPI blocked). Only the pure-Python tests (29) were run, with pandas/numpy on Python 3.13 and a stub logger.
- Not yet run: CatBoost/LightGBM training, Optuna, TimescaleDB schema creation, real downloads, WebSocket collectors, the real backtest, the Binance SDK adapter.
- The SDK adapter's response parsing is defensive; verify with a testnet smoke test (balance, position, entry, stop/TP, cancel, user stream).
- On-chain and Coinglass adapters are stubs. API BRICKS historical liquidation aggregates can be loaded through `backfill-liquidations`; historical long/short sides are unavailable from this source.
- No probability calibration; `orders` table is not yet populated; no Alembic migrations.
- Stage 5 (live trading) is intentionally unsupported.
