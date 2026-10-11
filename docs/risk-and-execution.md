# Risk and execution

## Hard limits (`risk/limits.py`)

Config can only go lower than these; `RiskCfg.clamped()` applies them on load and `plan_trade` clamps again independently.

| Limit | Max |
|---|---|
| Risk per trade | 2% of equity |
| Leverage | 5x (default config 3x) |
| Open positions | 3 |
| Daily loss limit | 10% |
| Weekly loss limit | 20% |
| ATR stop multiplier | at least 1.0 |

## Position planning (`risk/engine.py`)

1. Stop distance = `atr_stop_mult * ATR`.
2. Take-profit distance = `|pred_ret| * price`. The trade is rejected if reward/risk is below `min_reward_risk`.
3. Quantity = `equity * risk_per_trade / stop_distance`, then capped so notional ≤ `equity * leverage` (the cap reduces actual risk).
4. Rejections: invalid inputs, max positions reached, stop ≥ price, low reward/risk.

## Kill switch (`risk/kill_switch.py`)

Each cycle the live loop compares equity with the day-start (UTC) and week-start (ISO week) anchors. If a loss reaches the daily or weekly limit:

1. Write `state/KILL_SWITCH.lock` (before anything else).
2. Cancel all orders and close all positions at market.
3. Log CRITICAL and stop the loop.

The bot refuses to start while the lock exists. Clear it only with `trader-bot killswitch-reset`, after reviewing what happened. The code cannot revoke API keys: disable or rotate them on Binance manually.

## Order flow (`execution/manager.py`)

`open_trade`:
1. Round quantity down to the lot step and check min quantity/notional (skip the trade if invalid).
2. Set isolated margin and leverage.
3. Market entry.
4. Place a close-position `STOP_MARKET` and `TAKE_PROFIT_MARKET` through the Binance **Algo Order** endpoints (mark-price trigger).
5. If either protective order fails, immediately cancel everything and flatten. An unprotected position is never left open.

Per closed candle (`tick_bar`): if the exchange position is already flat (stop/TP fired), clean up; otherwise count bars held and close at the time exit (`horizon_candles`).

Startup (`reconcile`): remove stale book entries; flatten any position that has no protective orders or is not in the bot's own book.

## Live loop (`scheduler/runner.py`)

At each 4h boundary + 20s grace:
1. Backfill the latest candles/metrics for each symbol.
2. Read equity, run the kill-switch check.
3. `tick_bar` for each symbol.
4. For symbols with no open trade: build features, require the latest row to be the just-closed candle (stale data guard), load the promoted model, predict, make a signal, plan, trade. Every decision is written to `signals`.
5. Weekly: retrain and promote if not worse.

Errors in a cycle are logged and the loop continues. A user-data WebSocket stream tracks balance and order updates (logging; decisions use REST equity and positions).

## Security

- API keys only in `.env` (gitignored); `.env.example` is committed.
- Binance key settings: withdrawals disabled, IP whitelist, testnet keys for now.
- The CLI blocks live trading (`--live` or `BINANCE_TESTNET=false`).
