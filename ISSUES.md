# Known Issues

## Backtest completes with zero trades

**Status:** Diagnostics implemented; model/signal investigation remains open  
**Impact:** The current backtest report is not a useful profitability evaluation. It reports zero return, zero drawdown, and zero Sharpe because no positions were opened; `gate_passed` is false.

### Observed behavior

Running `uv run trader-bot backtest` completes and writes its report, but both BTCUSDT and ETHUSDT have `n_trades: 0`. The generated prediction files contain 6,594 rows per symbol for the configured backtest period.

Inspection of those predictions showed:

- BTCUSDT's top class was `flat` on 6,559 rows. Only 2 rows had a directional top class, confidence at least 0.75, and predicted-return sign agreeing with that direction.
- ETHUSDT's top class was `flat` on 6,590 rows. No row passed all three signal conditions.
- `trades.csv` contains only its header.

The follow-up diagnostics confirm the filters involved:

- BTCUSDT CV has 6,760 flat labels out of 7,967 OOS rows (84.8%), but predicts flat for 7,916 rows (99.4%). Directional recall is 0.36% for short and 0.15% for long; balanced accuracy is 0.333.
- BTCUSDT's backtest produced 2 signals after confidence and return-direction checks; both were rejected because predicted reward was below the configured minimum reward-to-risk ratio.
- ETHUSDT's backtest produced 4 directional top classes, but none met the confidence threshold.
- CV directional high-confidence coverage is 0.0% for BTCUSDT. The overall 84% accuracy is therefore dominated by flat-class predictions and does not indicate useful directional performance.

### Likely cause and investigation

The signal rule in `src/trader_bot/risk/engine.py` requires a non-flat top class, confidence at or above `model.confidence_threshold`, and agreement between the classifier direction and predicted-return sign. The risk engine can then reject a signal, including when its predicted reward is below the configured minimum reward-to-risk ratio. The current result is consistent with models that mostly predict the flat class, with additional filtering by signal and risk rules.

Training CV metrics now include per-fold train/test class counts, predicted class counts, per-class precision/recall, balanced accuracy, and directional-only confidence metrics. Backtest reports include prediction counts, counts passing each signal filter, risk-plan rejection reasons, and approved trades. The current run confirms both classifier flat-class collapse and the BTC reward/risk veto. Next evaluate class weighting and alternative label bands/horizons with purged walk-forward validation; separately evaluate changes to the reward/risk threshold. Keep the current settings as the baseline and do not select changes solely to force trades. Any candidate must be evaluated on held-out data with fees and slippage included.

An experiment command is now available: `uv run trader-bot experiment` compares ATR flat-label multipliers 0.5, 1.0, and 1.5 with unweighted and balanced classifiers using the existing purged CV folds. For separate backtest sensitivity runs, use `uv run trader-bot backtest --min-reward-risk 1.0 --out artifacts/backtest_rr1` or add `--balance-classes` and/or `--flat-atr-mult 0.5`. These experiments do not change defaults.

### Candidate experiment results (2026-10-04)

The best tested CV variant by balanced accuracy was `flat_atr_mult=0.5` with balanced classifiers: balanced accuracy 0.379 for BTCUSDT and 0.366 for ETHUSDT. Directional high-confidence precision was only 15.5% and 20.4%, respectively, at 1.95% and 1.23% coverage, so the classification improvement is weak.

Walk-forward backtests for that variant:

| Minimum reward/risk | Trades | Portfolio return | Sharpe | Profit factor |
|---:|---:|---:|---:|---:|
| 1.5 (current) | 6 | -0.78% | -0.365 | 0.539 |
| 1.0 | 23 | -0.05% | 0.028 | 0.990 |

Neither run passed the gate or showed a positive portfolio return after costs. Keep the production defaults unchanged; continue research rather than promoting this candidate.

### Resolution criteria

- Compare baseline against modeling and label variants using per-class metrics and trade counts across purged folds.
- Evaluate any selected model/signal changes on a separate held-out period with realistic costs.
- The resulting backtest has interpretable metrics and passes the configured gate before being treated as a viable strategy.
