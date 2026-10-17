# Known Issues

## Barrier mode, Round 1 (2026-10-17)

- **No barrier variant passed the Phase 11 rule.** The most folds with a positive mean R was 3 of 5,
  on both symbols. Phase 11 therefore stops here: **do not run `barrier-backtest --holdout-start`.**
- The best-looking cell (BTCUSDT, h=12, tp=4.0, +0.35 R on 542 signals) is concentrated in time:
  per-fold signals were 292, 0, 224, 26 and 0. ETHUSDT h=12 tp=4.0: 15, 0, 0, 819 and 22.
- AUC rises with `tp_atr_mult` (0.62–0.66) mainly because wide targets have low base rates (2–12%).
  Higher AUC alone does not mean the model is more useful.
- Costs are about 0.07 R per round trip at typical ATR, and `always_long_mean_net_r` is about zero
  or negative in every cell. The model has to clear this cost before it shows any edge.
- `n_signals` counts **overlapping** rows (a signal every bar during a 12-bar hold). The effective
  trade count is much smaller, so these averages look more certain than they are.
- Baselines: all four rules are negative or flat on the holdout. `breakout_20` was the only positive
  one on validation (Sharpe 0.33, 90% CI [-0.73, 1.44]) and lost on the holdout (Sharpe -1.52).
- The 4h history already starts around 2019-09 (≈13.2k rows per symbol), so backlog **H1 is done**.
  More rows per symbol are not available; the remaining data lever is pooling symbols.

Status: no candidate passed; holdout not used for barrier mode.

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

### Barrier research checkpoint (2026-10-17)

The phase 11 protocol was run in order. Reports:

- Baselines: `artifacts/baselines/report.json`
- Pre-holdout barrier experiment: `artifacts/experiments/barrier-20261017T095437Z/report.json`

The baseline strategies did not pass the backtest gate. `breakout_20` was modestly positive on validation (+3.86% return, Sharpe 0.325) but negative on holdout (-9.22%, Sharpe -1.523). `funding_contrarian` was nearly flat on holdout (+0.04%, Sharpe 0.028); always-long and momentum were negative on both segments.

No barrier candidate passed the pre-holdout selection rule. Although several BTCUSDT variants exceeded their always-long mean net R, the highest `folds_with_positive_mean_r` was 3 of 5. ETHUSDT also peaked at 3 of 5, below the required 4, so no candidate qualifies across both symbols. The strongest BTCUSDT mean net R was 0.351 at horizon 12 / TP 4 ATR, but it passed only 2 folds; the strongest ETHUSDT mean net R among risk-floor candidates was 0.049 at horizon 12 / TP 4 ATR, also only 2 folds. The barrier holdout backtest was therefore not run, as required by the protocol. Do not promote or enable barrier predictions from these results.

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

## Barrier mode, Round 2 (2026-10-17)

Ran the fixed Phase 18 sequence with the prescribed grid (`horizons=6,12,18`, `tp_atr_mults=3.0,4.0`). This is 24 selection cells per symbol across R2-A through R2-D. R2-W is the report-only lower-fee what-if. No holdout backtest was run. R2-W used the R2-A configuration because R2-A's best cell (h=18, TP=4.0) had the highest count of the 12 required per-symbol gates (9/12); no cell passed all gates. The fee what-if is excluded from selection.

R2-A through R2-D all failed the joint selection rule. No cell met all six criteria for both symbols, including at least four positive non-overlapping folds and t-statistic ≥2.5. Some cells were positive on average, but results were inconsistent across folds/symbols and/or did not clear the significance and baseline criteria.

### R2-A — {'feature_groups': {'price': True, 'derivatives': True, 'liquidations': False, 'onchain': False, 'cross_asset': False, 'calendar': False, 'derivs_extra': False}, 'model_size': 'default', 'taker_fee': 0.0005, 'slippage_bps': 2.0}

| Horizon | TP ATR | BTC mean R | BTC t | BTC + folds/5 | BTC trades | ETH mean R | ETH t | ETH + folds/5 | ETH trades | Gates met/12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 3.0 | -0.059 | -0.86 | 2/5 | 143 | -0.013 | -0.22 | 0/5 | 201 | 6/12 |
| 6 | 4.0 | +0.154 | 1.31 | 2/5 | 63 | -0.023 | -0.19 | 2/5 | 58 | 4/12 |
| 12 | 3.0 | -0.064 | -0.52 | 2/5 | 70 | +0.052 | 0.57 | 3/5 | 122 | 7/12 |
| 12 | 4.0 | +0.224 | 2.07 | 3/5 | 111 | +0.039 | 0.39 | 2/5 | 121 | 6/12 |
| 18 | 3.0 | -0.008 | -0.10 | 1/5 | 151 | +0.025 | 0.28 | 2/5 | 139 | 7/12 |
| 18 | 4.0 | +0.258 | 2.72 | 4/5 | 183 | -0.002 | -0.02 | 3/5 | 189 | 9/12 |

### R2-B — {'feature_groups': {'price': True, 'derivatives': True, 'liquidations': False, 'onchain': False, 'cross_asset': False, 'calendar': False, 'derivs_extra': False}, 'model_size': 'default', 'taker_fee': 0.0005, 'slippage_bps': 2.0}

| Horizon | TP ATR | BTC mean R | BTC t | BTC + folds/5 | BTC trades | ETH mean R | ETH t | ETH + folds/5 | ETH trades | Gates met/12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 3.0 | -0.127 | -1.28 | 1/5 | 79 | +0.033 | 0.18 | 2/5 | 32 | 5/12 |
| 6 | 4.0 | -0.208 | -2.48 | 0/5 | 106 | -0.086 | -0.65 | 1/5 | 46 | 3/12 |
| 12 | 3.0 | -0.108 | -1.03 | 1/5 | 96 | +0.018 | 0.15 | 2/5 | 70 | 7/12 |
| 12 | 4.0 | -0.271 | -2.18 | 0/5 | 66 | -0.112 | -0.82 | 1/5 | 65 | 4/12 |
| 18 | 3.0 | +0.011 | 0.09 | 3/5 | 91 | -0.017 | -0.14 | 2/5 | 78 | 7/12 |
| 18 | 4.0 | -0.145 | -1.16 | 2/5 | 94 | +0.092 | 0.76 | 2/5 | 113 | 7/12 |

### R2-C — {'feature_groups': {'price': True, 'derivatives': True, 'liquidations': False, 'onchain': False, 'cross_asset': True, 'calendar': True, 'derivs_extra': True}, 'model_size': 'default', 'taker_fee': 0.0005, 'slippage_bps': 2.0}

| Horizon | TP ATR | BTC mean R | BTC t | BTC + folds/5 | BTC trades | ETH mean R | ETH t | ETH + folds/5 | ETH trades | Gates met/12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 3.0 | -0.109 | -1.08 | 1/5 | 80 | +0.080 | 0.47 | 2/5 | 34 | 4/12 |
| 6 | 4.0 | -0.084 | -0.70 | 1/5 | 60 | +0.105 | 0.34 | 2/5 | 12 | 4/12 |
| 12 | 3.0 | -0.210 | -1.87 | 1/5 | 85 | -0.299 | -2.32 | 0/5 | 67 | 6/12 |
| 12 | 4.0 | -0.138 | -1.11 | 0/5 | 74 | -0.035 | -0.21 | 1/5 | 50 | 3/12 |
| 18 | 3.0 | -0.098 | -0.85 | 2/5 | 95 | +0.087 | 0.57 | 3/5 | 58 | 6/12 |
| 18 | 4.0 | +0.169 | 1.14 | 3/5 | 77 | -0.094 | -0.59 | 1/5 | 64 | 7/12 |

### R2-D — {'feature_groups': {'price': True, 'derivatives': True, 'liquidations': False, 'onchain': False, 'cross_asset': True, 'calendar': True, 'derivs_extra': True}, 'model_size': 'small', 'taker_fee': 0.0005, 'slippage_bps': 2.0}

| Horizon | TP ATR | BTC mean R | BTC t | BTC + folds/5 | BTC trades | ETH mean R | ETH t | ETH + folds/5 | ETH trades | Gates met/12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 3.0 | -0.070 | -0.85 | 1/5 | 122 | +0.096 | 0.82 | 2/5 | 70 | 7/12 |
| 6 | 4.0 | -0.100 | -0.90 | 1/5 | 82 | -0.083 | -0.53 | 1/5 | 41 | 3/12 |
| 12 | 3.0 | -0.170 | -1.63 | 1/5 | 101 | -0.168 | -1.69 | 1/5 | 106 | 6/12 |
| 12 | 4.0 | -0.182 | -1.24 | 0/5 | 53 | -0.241 | -1.38 | 0/5 | 42 | 2/12 |
| 18 | 3.0 | +0.033 | 0.28 | 2/5 | 94 | -0.189 | -1.61 | 1/5 | 83 | 7/12 |
| 18 | 4.0 | +0.077 | 0.54 | 3/5 | 81 | -0.167 | -1.06 | 2/5 | 63 | 7/12 |

### R2-W — {'feature_groups': {'price': True, 'derivatives': True, 'liquidations': False, 'onchain': False, 'cross_asset': False, 'calendar': False, 'derivs_extra': False}, 'model_size': 'default', 'taker_fee': 0.0002, 'slippage_bps': 2.0}

| Horizon | TP ATR | BTC mean R | BTC t | BTC + folds/5 | BTC trades | ETH mean R | ETH t | ETH + folds/5 | ETH trades | Gates met/12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 3.0 | -0.047 | -0.70 | 1/5 | 153 | +0.007 | 0.13 | 2/5 | 201 | 7/12 |
| 6 | 4.0 | +0.177 | 1.52 | 2/5 | 64 | +0.012 | 0.10 | 2/5 | 59 | 5/12 |
| 12 | 3.0 | -0.044 | -0.37 | 1/5 | 75 | +0.094 | 1.05 | 4/5 | 124 | 8/12 |
| 12 | 4.0 | +0.265 | 2.44 | 2/5 | 112 | +0.058 | 0.57 | 2/5 | 121 | 6/12 |
| 18 | 3.0 | -0.019 | -0.23 | 1/5 | 161 | +0.056 | 0.63 | 3/5 | 142 | 7/12 |
| 18 | 4.0 | +0.260 | 2.75 | 4/5 | 185 | +0.015 | 0.17 | 3/5 | 189 | 9/12 |


Conclusion: no robust 4h directional edge after costs with available data. Phase 18 stops here; do not run `barrier-backtest --holdout-start`. Phase 19 is gated on a qualifying holdout result and explicit approval, so live wiring was not started.
