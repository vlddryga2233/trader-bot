# Strategy Readiness Gaps

## Summary

The repository has a substantial ML, backtesting, risk, and execution foundation, but the available results do not demonstrate a durable trading edge after costs. Treat the strategy as experimental; the current evidence does not support live futures trading.

## Evidence in the Repository

- The saved default backtest has no approved trades: the signal filters and reward/risk check reject the candidate entries.
- A more permissive saved run produced 23 trades with a slightly negative total return and a Sharpe ratio close to zero. This is too few trades to draw a reliable conclusion.
- The saved ML experiment reports low directional signal coverage and weak directional precision. High overall accuracy is largely driven by the flat class and does not establish useful long/short predictions.
- The documented backtest already uses walk-forward predictions, taker fees, slippage, funding payments, conservative stop-before-target ordering, and risk limits. Those are useful foundations, but they do not resolve the poor strategy results.

## Gaps to Close

1. **Rule-based baselines:** There is no documented comparison against simple rule-based strategies under the same symbols, dates, sizing, and cost assumptions. Without this, the incremental value of ML is unknown.
2. **Untouched final evaluation:** Strategy and threshold selection need to be separated from a final holdout period. Repeatedly tuning against the same backtest risks selecting for historical noise.
3. **Futures simulation fidelity:** The backtest models fees, slippage, and funding, but the available docs do not show liquidation or maintenance-margin-tier simulation, or validation of exchange trigger and fill behavior. Binance futures liquidation depends on mark price and maintenance margin; those mechanics should be reflected in risk checks and simulation.
4. **Combined portfolio risk:** BTC and ETH exposure is assessed as independent books in the documented backtest. Risk should also be evaluated across the combined account, including correlated positions and unrealized P&L.
5. **Execution and recovery validation:** The docs describe reconciliation and a kill switch, but test evidence is needed for partial fills, disconnects, rejected or missing protective orders, and restart recovery.

## Recommended Next Steps

1. Add simple rule-based baselines and compare them with ML using identical walk-forward periods, sizing, and trading costs.
2. Diagnose weak or rare ML signals and unprofitable trades. Do not loosen thresholds solely to increase trade count.
3. Select the approach without using a final holdout, then evaluate that holdout once. Report trade count, net return, drawdown, costs, and results by symbol and time period.
4. Improve or validate the futures simulator for mark-price triggers, liquidation and margin behavior, and realistic fills.
5. Verify exchange execution and recovery behavior on testnet, then paper trade the full loop before considering live use.

## Existing Strengths

- Features use trailing-only calculations and as-of joins to avoid look-ahead.
- Model evaluation uses purged walk-forward validation, and the backtest generates out-of-sample predictions.
- The execution design includes protective orders, startup reconciliation, hard risk limits, and a kill switch.
- The backtest includes fees, slippage, funding, and conservative stop handling.

These controls improve process safety; they do not establish profitability. Reassess this document as new baseline, holdout, simulator, and testnet evidence is added.
