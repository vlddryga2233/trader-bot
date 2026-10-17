# Model Improvement Implementation Plan

This plan is written so a low-cost coding model can follow it step by step. Every phase lists the
files to touch, the code to write, the tests to add, and the commands that prove the phase is done.

**Do the phases in order. Do not skip a verification step. Do not change anything not listed.**

---

## Why this plan exists

Review findings (see also `GAPS.md`, `ISSUES.md`):

1. **Take-profit sizing structurally vetoes trades.** `risk/engine.py:63-68` sets stop = 2×ATR(4h)
   and take-profit = `|pred_ret| × price`. With `min_reward_risk: 1.5`, TP must be ≥ 3 ATR, but a
   typical 24h move is ≈ ATR×√6 ≈ 2.45 ATR and an RMSE regressor shrinks `pred_ret` towards 0.
2. **Label ≠ traded outcome.** Labels are close-to-close returns at t+6 (`labeling/labels.py`);
   the trade result depends on whether the stop, TP or time exit is hit first.
3. **3-class argmax + 0.75 threshold on uncalibrated, averaged probabilities** collapses to "flat".
4. **Tuned params are not what gets backtested or retrained live** (`cli.py` backtest passes no
   params; `scheduler/runner.py:104` retrains with defaults).
5. **TP exits have no slippage** in `backtest/engine.py:140`.
6. **No baselines and no confidence intervals** for Sharpe.

The fix: add an **opt-in "barrier" mode** (default stays `classic`) that
- labels each bar with the *exact* stop / take-profit / time-exit outcome (triple barrier),
- trains two calibrated binary classifiers: P(long TP hit first), P(short TP hit first),
- trades only when the **expected value in R units, after costs**, is positive enough,
- uses a fixed take-profit of `tp_atr_mult × ATR`, so reward/risk is known in advance.

---

## Ground rules (read before starting)

- Python 3.12, 4-space indent, Ruff line length 100. Match existing style.
- **Never change existing defaults** in `config/strategy.yaml` or `config/settings.py`. New
  behaviour must be opt-in (`signal_mode: barrier`).
- **Never touch** `risk/limits.py`, `risk/kill_switch.py`, `execution/*`. Live trading stays blocked.
- Tests must use synthetic data only. No database, no network, no API keys.
- After **every** phase run:
  ```bash
  uv run pytest -q
  uv run ruff check src tests
  ```
  Both must pass before the next phase. If `uv` cannot reach PyPI, run `uv sync` once when
  network is available; `.venv` currently lacks `pytest`/`ruff`.
- Commit after each phase with the message given in that phase, plus the trailer:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
- If a step is ambiguous or a test fails twice after a fix attempt, **stop and ask a human**.

---

## Round 1 — done (Phases 0–11, commit `44b5598`)

Implemented and merged; do not redo. Use these as building blocks in Round 2.

| Phase | Result | Main files |
|---|---|---|
| 1 | TP-exit slippage; tuned params reused by retrain; `backtest --registry-params` | `backtest/engine.py`, `models/retrain.py`, `models/registry.py`, `cli.py` |
| 2 | `ctx_high` / `ctx_low` context columns | `features/builder.py` |
| 3 | Triple-barrier labels (stop checked first, R units) | `labeling/barriers.py` |
| 4 | `signal_mode` (default `classic`) and `BarrierCfg` | `config/settings.py`, `config/strategy.yaml` |
| 5 | EV rule `EV = p·RR − (1−p) − cost_R` | `risk/engine.py` (`round_trip_cost`, `make_ev_signal`) |
| 6 | Pluggable `signal_fn` in the backtester | `backtest/engine.py` |
| 7 | `BarrierDataset`, `BarrierModel` (calibrated), `cv_evaluate_barrier`, walk-forward predictions | `models/barrier.py` |
| 8 | `bootstrap_sharpe_ci`; `barrier-experiment`, `barrier-backtest` | `backtest/metrics.py`, `backtest/segments.py`, `backtest/run_barrier.py`, `models/barrier_experiment.py` |
| 9 | Four rule baselines, `baselines` command | `backtest/baselines.py` |
| 10 | `docs/data-and-ml.md`, `docs/architecture.md` updated | README and `GAPS.md` still pending → Phase 12.3 |
| 11 | Research protocol run: **no candidate passed**; holdout not used | results below, recording → Phase 12.1 |

Round 1 selection rule (for reference): `rr_meets_risk_floor` and `folds_with_positive_mean_r >= 4`
for both symbols, `n_signals >= 50`, beats `always_long_mean_net_r`.

---

# Round 2 — after the first barrier experiment (2026-10-17)

## What Round 1 showed

Source: `artifacts/experiments/barrier-20261017T095437Z/report.json`, `artifacts/baselines/report.json`.

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

## Round 2 goals, in order

1. **Measure honestly** (Phase 13): non-overlapping trades, t-statistics, per-fold signal counts.
2. **More training data** (Phase 15): train one pooled BTC+ETH model.
3. **New information** (Phase 14): cross-asset, calendar, funding-z, OI-price divergence and
   stationary derivatives features, each behind its own flag.
4. **Less overfitting** (Phase 16): a smaller model option.
5. **Cost what-if** (Phase 17): rerun with maker fees to see whether an edge exists at lower cost
   (report only; execution does not change).
6. **One** final experiment (Phase 18) with a fixed variant budget, then the holdout at most once.

Ground rules, verification commands and commit trailer are the same as Round 1. Every new flag
defaults to **off**, so classic and Round 1 outputs stay unchanged.

---

## Phase 12 — Housekeeping

### 12.1 Record the Round 1 result

Append to `ISSUES.md` a section `## Barrier mode, Round 1 (2026-10-17)` containing the bullet list
under "What Round 1 showed" above (copy it as is). End with:
`Status: no candidate passed; holdout not used for barrier mode.`

### 12.2 Stop committing generated artifacts

`.gitignore`: replace the line `artifacts/models` with:

```gitignore
artifacts/
```

Then untrack the files without deleting them locally:

```bash
git rm -r --cached artifacts
```

Do **not** delete the local `artifacts/` directory; Round 2 compares against it.

### 12.3 Finish the Round 1 documentation

- `README.md` Quick start: after the `backtest` line add
  ```bash
  uv run trader-bot baselines --holdout-start 2025-10-01          # research
  uv run trader-bot barrier-experiment --holdout-start 2025-10-01 # research
  uv run trader-bot barrier-backtest --holdout-start 2025-10-01   # research, run once
  ```
- `GAPS.md`, "Gaps to Close" item 1: append the sentence
  `` `trader-bot baselines` now runs four rule-based baselines with the same simulator and costs; none is profitable on the 2025-10 holdout. ``

Commit: `Record barrier round 1 result; stop tracking artifacts; finish docs`

---

## Phase 13 — Honest signal metrics (non-overlapping trades)

File: `src/trader_bot/models/barrier.py`

1. Add a helper above `barrier_signal_metrics`:

```python
def non_overlapping(signal: np.ndarray, horizon: int) -> np.ndarray:
    """Greedy mask: take a signal, then skip the next horizon-1 rows (one position at a time)."""
    taken = np.zeros(len(signal), dtype=bool)
    next_free = 0
    for i in np.flatnonzero(signal):
        if i >= next_free:
            taken[i] = True
            next_free = i + horizon
    return taken


def _t_stat(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    sd = values.std(ddof=1)
    return float(values.mean() / (sd / np.sqrt(len(values)))) if sd > 0 else None
```

2. In `barrier_signal_metrics`, before `return`, add:

```python
    trades = non_overlapping(signal, cfg.labeling.horizon_candles)
    trade_r = realized[trades]
```

and add these keys to the returned dict:

```python
        "n_trades_nonoverlap": int(trades.sum()),
        "mean_net_r_nonoverlap": float(trade_r.mean()) if len(trade_r) else None,
        "t_stat_nonoverlap": _t_stat(trade_r),
```

Note: `barrier_signal_metrics` receives a single symbol's rows in time order, so a positional
mask is correct. Phase 15 calls it once per symbol for the same reason.

3. In `cv_evaluate_barrier`, add `"n_trades_nonoverlap"` and `"mean_net_r_nonoverlap"` to the
   tuple of per-fold keys. Add this helper below `cv_evaluate_barrier` (Phase 15 reuses it):

```python
FOLD_KEYS = ("auc_long", "auc_short", "n_signals", "hit_rate_on_signals",
             "mean_net_r_on_signals", "n_trades_nonoverlap", "mean_net_r_nonoverlap")


def summarize_folds(metrics: dict, folds: list[dict]) -> dict:
    def positive(key: str) -> int:
        return sum(1 for fold in folds if fold[key] is not None and fold[key] > 0)

    metrics["cv_folds"] = folds
    metrics["folds_with_positive_mean_r"] = positive("mean_net_r_on_signals")
    metrics["folds_with_positive_nonoverlap_r"] = positive("mean_net_r_nonoverlap")
    metrics["folds_with_signals"] = sum(1 for fold in folds if fold["n_signals"] > 0)
    return metrics
```

   In `cv_evaluate_barrier`, use `**{key: fold_metrics[key] for key in FOLD_KEYS}` for the
   per-fold dict, and replace everything from `metrics["cv_folds"] = folds` to `return metrics`
   with `return summarize_folds(metrics, folds)`.

4. File `src/trader_bot/models/barrier_experiment.py`: extend `SUMMARY_KEYS` with
   `"n_trades_nonoverlap", "mean_net_r_nonoverlap", "t_stat_nonoverlap",
   "folds_with_positive_nonoverlap_r", "folds_with_signals"`.

Test — add to `tests/test_barrier_model.py`:

```python
from trader_bot.models.barrier import _t_stat, non_overlapping
```

```python
def test_non_overlapping_skips_rows_inside_holding_period():
    signal = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1], dtype=bool)
    taken = non_overlapping(signal, horizon=3)
    assert np.flatnonzero(taken).tolist() == [0, 4, 8]


def test_non_overlapping_horizon_one_keeps_all():
    signal = np.array([1, 0, 1, 1], dtype=bool)
    assert non_overlapping(signal, 1).tolist() == signal.tolist()


def test_t_stat():
    assert _t_stat(np.array([1.0])) is None
    assert _t_stat(np.array([1.0, 1.0])) is None
    assert _t_stat(np.array([1.0, 2.0, 3.0])) > 3
```

Commit: `Report non-overlapping trade metrics in barrier CV`

---

## Phase 14 — New feature groups (all default off)

### 14.1 Config flags

File: `src/trader_bot/config/settings.py`, class `FeatureGroups`, add:

```python
    cross_asset: bool = False
    calendar: bool = False
    derivs_extra: bool = False
```

File: `config/strategy.yaml`, under `features.groups`, add:

```yaml
    cross_asset: false    # reference-symbol returns/volatility (BTC for ETH, ETH for BTC)
    calendar: false       # decision hour and weekday
    derivs_extra: false   # funding z-score, OI-price divergence, stationary OI/long-short
```

### 14.2 Builder functions

File: `src/trader_bot/features/builder.py`. Add these functions above `build_features`:

```python
def cross_asset_features(base: pd.DataFrame, ref_k4: pd.DataFrame) -> pd.DataFrame:
    """Reference-symbol 4h returns/volatility, available only after the reference candle closes."""
    if ref_k4.empty:
        return base
    lc = np.log(ref_k4["close"])
    r = pd.DataFrame(index=ref_k4.index)
    for n in (1, 6, 42):
        r[f"ref_ret_{n}"] = lc.diff(n)
    r["ref_vol_42"] = lc.diff().rolling(42).std()
    out = _asof(base, _stamp_closed(r, PRIMARY), tolerance=PRIMARY)
    out["rel_ret_42"] = out["ret_42"] - out["ref_ret_42"]
    return out


def calendar_features(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["cal_hour"] = out.index.hour
    out["cal_dow"] = out.index.dayofweek
    return out


def derivs_extra_features(base: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Needs the derivatives group's columns; computed on the 4h decision grid, trailing only."""
    out = base
    if not funding.empty:
        rate = funding["rate"]
        z = (rate - rate.rolling(90).mean()) / rate.rolling(90).std()  # 90 x 8h = 30 days
        out = _asof(out, pd.DataFrame({"funding_z_30d": z}))
    out = out.copy()
    if "oi_chg_24h" in out:
        out["oi_px_div_24h"] = out["oi_chg_24h"] - (np.exp(out["ret_6"]) - 1)
    for col in ("oi_value", "top_ls_accounts", "top_ls_positions", "global_ls"):
        if col in out:
            mean = out[col].rolling(180, min_periods=90).mean()
            std = out[col].rolling(180, min_periods=90).std()
            out[f"{col}_z_30d"] = (out[col] - mean) / std.replace(0, np.nan)
    return out
```

Change the `build_features` signature by adding a keyword argument at the end:

```python
    ref_k4: pd.DataFrame | None = None,
```

and, after the onchain block and before `feats.index.name = "decision_time"`, add:

```python
    if groups.derivatives and groups.derivs_extra:
        feats = derivs_extra_features(feats, funding)
    if groups.cross_asset and ref_k4 is not None:
        feats = cross_asset_features(feats, ref_k4)
    if groups.calendar:
        feats = calendar_features(feats)
```

`calendar_features` must run last: it reads the final index, which is the decision time.
`_asof` keeps the index name, so `out.index` is still the decision time.

### 14.3 Pipeline: load the reference symbol

File: `src/trader_bot/features/pipeline.py`:

```python
def reference_symbol(symbol: str) -> str:
    return "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"
```

In `features_for_symbol`, before `build_features`, add:

```python
    ref_k4 = None
    if cfg.features.groups.cross_asset:
        ref_k4 = store.load_frame(engine, schema.klines, "open_time", reference_symbol(symbol),
                                  extra={"interval": "4h"})
```

and pass `ref_k4=ref_k4` as the last argument to `build_features`.

### 14.4 Tests

Add to `tests/test_features.py`:

```python
def _build_all(k4, k1, kd, funding, metrics, liq, onchain, ref_k4):
    groups = FeatureGroups(price=True, derivatives=True, cross_asset=True, calendar=True,
                           derivs_extra=True)
    return build_features(k4, k1, kd, funding, metrics, liq, onchain, groups, ref_k4=ref_k4)


def test_new_feature_groups_present_and_leak_free():
    inputs = _inputs()
    ref = _klines("4h", N, 9)
    full = _build_all(*inputs, ref)
    for col in ("ref_ret_42", "rel_ret_42", "cal_hour", "cal_dow", "funding_z_30d",
                "oi_px_div_24h", "oi_value_z_30d"):
        assert col in full.columns
    t = full.index[300]
    k4, k1, kd, funding, metrics, liq, oc = inputs
    cut = _build_all(
        k4[k4.index + pd.Timedelta(hours=4) <= t],
        k1[k1.index + pd.Timedelta(hours=1) <= t],
        kd[kd.index + pd.Timedelta(days=1) <= t],
        funding[funding.index <= t], metrics[metrics.index <= t], liq, oc,
        ref[ref.index + pd.Timedelta(hours=4) <= t],
    )
    cols = model_columns(full)
    pd.testing.assert_series_equal(full.loc[t, cols], cut.loc[t, cols], check_names=False)


def test_new_groups_off_by_default_keep_columns_unchanged():
    assert "cal_hour" not in _build(*_inputs()).columns
```

Note: `test_new_feature_groups_present_and_leak_free` reuses the truncation pattern from
`test_no_future_leakage_when_future_is_truncated`. If that test truncates `funding` or
`metrics` differently, copy its exact truncation lines instead of the ones above.

Commit: `Add cross-asset, calendar and extra derivatives feature groups`

---

## Phase 15 — Pooled BTC+ETH barrier model

One model trained on both symbols roughly doubles the training rows. `sym_is_eth` lets it
learn per-symbol differences.

### 15.1 Code

Create `src/trader_bot/models/pooled.py`:

```python
"""Pool several symbols' barrier datasets into one time-ordered training set."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader_bot.config.settings import StrategyConfig
from trader_bot.models.barrier import (
    FOLD_KEYS, BarrierDataset, BarrierModel, barrier_signal_metrics, summarize_folds,
)
from trader_bot.models.cv import purged_walk_forward

SYMBOL_COL = "sym_id"


def pool_datasets(datasets: dict[str, BarrierDataset]) -> tuple[BarrierDataset, pd.Series]:
    """Return the pooled labeled dataset (sorted by time, then symbol) and a symbol Series."""
    frames, symbols = [], list(datasets)
    for code, symbol in enumerate(symbols):
        ds = datasets[symbol]
        lab = ds.has_label.to_numpy(dtype=bool)
        frame = ds.X[lab].copy()
        frame[SYMBOL_COL] = code
        frame["_y_long"] = ds.y_long[lab].to_numpy()
        frame["_y_short"] = ds.y_short[lab].to_numpy()
        frame["_r_long"] = ds.r_long[lab].to_numpy()
        frame["_r_short"] = ds.r_short[lab].to_numpy()
        frame["_symbol"] = symbol
        frames.append(frame.reset_index())
    time_col = frames[0].columns[0]
    pooled = pd.concat(frames, ignore_index=True).sort_values([time_col, SYMBOL_COL])
    pooled = pooled.set_index(time_col)
    columns = sorted(set.intersection(*(set(datasets[s].columns) for s in symbols)))
    columns.append(SYMBOL_COL)
    out = BarrierDataset(
        pooled.drop(columns=["_y_long", "_y_short", "_r_long", "_r_short", "_symbol"]),
        pooled["_y_long"], pooled["_y_short"], pooled["_r_long"], pooled["_r_short"],
        pd.Series(True, index=pooled.index), columns,
    )
    return out, pooled["_symbol"]


def cv_evaluate_pooled(datasets: dict[str, BarrierDataset], cfg: StrategyConfig) -> dict:
    """Purged CV on the pooled set; metrics are reported per symbol from pooled OOS predictions.

    Rows are interleaved (one per symbol per timestamp), so purge/embargo/min_train are scaled
    by the number of symbols to keep the same time gaps as single-symbol CV.
    """
    ds, symbols = pool_datasets(datasets)
    sym = symbols.to_numpy()
    k = len(datasets)
    m, h = cfg.model, cfg.labeling.horizon_candles
    X = ds.X

    def score(P: pd.DataFrame, rows: np.ndarray, symbol: str) -> dict:
        mask = sym[rows] == symbol
        r = rows[mask]
        return barrier_signal_metrics(
            P[mask], X.iloc[r], ds.y_long.iloc[r], ds.y_short.iloc[r],
            ds.r_long.iloc[r], ds.r_short.iloc[r], cfg,
        )

    preds, all_rows = [], []
    folds: dict[str, list[dict]] = {symbol: [] for symbol in datasets}
    for fold, (tr, te) in enumerate(
        purged_walk_forward(len(X), m.cv_splits, h * k, m.embargo_candles * k,
                            m.min_train_candles * k),
        start=1,
    ):
        model = BarrierModel(ds.columns, purge=(h + m.embargo_candles) * k,
                             calib_frac=cfg.barrier.calib_frac).fit(
            X.iloc[tr], ds.y_long.iloc[tr], ds.y_short.iloc[tr]
        )
        P = model.predict(X.iloc[te])
        preds.append(P)
        all_rows.append(te)
        for symbol in datasets:
            fm = score(P, te, symbol)
            folds[symbol].append({"fold": fold, "n_test": fm["n_oos"],
                                  **{key: fm[key] for key in FOLD_KEYS}})
    P, rows = pd.concat(preds), np.concatenate(all_rows)
    return {symbol: summarize_folds(score(P, rows, symbol), folds[symbol])
            for symbol in datasets}
```

Note: `purged_walk_forward`'s train cut is a row gap. With k interleaved symbols, a gap of
`(h + embargo) * k` rows is `(h + embargo)` candles in time, the same as single-symbol CV.

### 15.2 Experiment hook

File: `src/trader_bot/models/barrier_experiment.py`:

1. Add a parameter `pooled: bool = False` to `run_barrier_experiment`.
2. If `pooled` is true, build every symbol's features first and, for each `(horizon, tp)`,
   build each symbol's `build_barrier_dataset(feats, variant)` and call
   `cv_evaluate_pooled(datasets, variant)`. Store the result under
   `report["symbols"][symbol]` in the same `{"horizon_candles", "tp_atr_mult", "metrics"}`
   shape, so the CLI summary works unchanged. Add `"pooled": True` to `report`.
3. Leave the existing per-symbol loop as the `else` branch, unchanged.

File: `src/trader_bot/cli.py`, `barrier_experiment`: add
`pooled: bool = typer.Option(False, "--pooled", help="Train one model on all symbols")` and pass
`pooled=pooled`.

### 15.3 Test

Add to `tests/test_barrier_model.py`:

```python
def test_pool_datasets_interleaves_symbols_in_time_order():
    from trader_bot.models.pooled import SYMBOL_COL, pool_datasets

    a, b = _synthetic(n=50, seed=1), _synthetic(n=50, seed=2)
    pooled, symbols = pool_datasets({"BTCUSDT": a, "ETHUSDT": b})
    assert len(pooled.X) == 100
    assert pooled.X.index.is_monotonic_increasing
    assert symbols.iloc[:2].tolist() == ["BTCUSDT", "ETHUSDT"]
    assert SYMBOL_COL in pooled.columns


def test_cv_evaluate_pooled_reports_each_symbol():
    from trader_bot.models.pooled import cv_evaluate_pooled

    cfg = StrategyConfig()
    cfg.model = cfg.model.model_copy(update={"min_train_candles": 300, "cv_splits": 2})
    out = cv_evaluate_pooled({"BTCUSDT": _synthetic(n=800, seed=1),
                              "ETHUSDT": _synthetic(n=800, seed=2)}, cfg)
    assert set(out) == {"BTCUSDT", "ETHUSDT"}
    assert out["BTCUSDT"]["auc_long"] > 0.6
```

Commit: `Add pooled multi-symbol barrier model to experiments`

---

## Phase 16 — Smaller-model option

File: `src/trader_bot/config/settings.py`, `BarrierCfg`, add:

```python
    model_size: Literal["default", "small"] = "default"
```

File: `src/trader_bot/models/barrier.py`:

```python
SMALL_PARAMS = {
    "lgbm": {"num_leaves": 15, "min_child_samples": 100, "n_estimators": 300},
    "catboost": {"depth": 4, "l2_leaf_reg": 5.0},
}
```

Change `_binary_models(seed)` to `_binary_models(seed, size="default")`, merging
`SMALL_PARAMS[...]` over `DEFAULT_PARAMS[...]` when `size == "small"`. Add `size: str = "default"`
to `BarrierModel.__init__` (stored as `self.size`) and pass it in `fit`. Pass
`size=cfg.barrier.model_size` everywhere a `BarrierModel` is created (`cv_evaluate_barrier`,
`walk_forward_barrier_predictions`, `models/pooled.py`).

File: `config/strategy.yaml`, under `barrier:` add `model_size: default   # or "small"`.

Test in `tests/test_barrier_model.py`:

```python
def test_small_model_option_fits():
    ds = _synthetic()
    model = BarrierModel(ds.columns, purge=12, calib_frac=0.3, size="small").fit(
        ds.X, ds.y_long, ds.y_short)
    assert model.models["long"][0].get_params()["num_leaves"] == 15
```

Commit: `Add small-model option for barrier classifiers`

---

## Phase 17 — Experiment CLI options (no new logic)

File: `src/trader_bot/cli.py`, `barrier_experiment`, add options and apply them to `cfg`
before calling `run_barrier_experiment`:

```python
    feature_groups: str = typer.Option(
        "", help="Comma-separated extra feature groups to enable: cross_asset,calendar,derivs_extra"
    ),
    model_size: str = typer.Option("default", help="default or small"),
    taker_fee: float | None = typer.Option(
        None, help="What-if fee for cost sensitivity (report only; never used for selection)"
    ),
```

```python
    allowed = {"cross_asset", "calendar", "derivs_extra"}
    extra = {g.strip() for g in feature_groups.split(",") if g.strip()}
    if not extra <= allowed:
        raise typer.BadParameter(f"feature groups must be in {sorted(allowed)}")
    if model_size not in ("default", "small"):
        raise typer.BadParameter("model_size must be default or small")
    groups = cfg.features.groups.model_copy(update={g: True for g in extra})
    cfg.features = cfg.features.model_copy(update={"groups": groups})
    cfg.barrier = cfg.barrier.model_copy(update={"model_size": model_size})
    if taker_fee is not None:
        cfg.backtest = cfg.backtest.model_copy(update={"taker_fee": taker_fee})
```

File: `src/trader_bot/models/barrier_experiment.py`: record the settings in the report:

```python
    report["settings"] = {
        "feature_groups": cfg.features.groups.model_dump(),
        "model_size": cfg.barrier.model_size,
        "taker_fee": cfg.backtest.taker_fee,
        "slippage_bps": cfg.backtest.slippage_bps,
    }
```

Verify: `uv run trader-bot barrier-experiment --help` lists `--pooled`, `--feature-groups`,
`--model-size`, `--taker-fee`.

Commit: `Expose feature groups, model size and fee what-if in barrier-experiment`

---

## Phase 18 — Round 2 research protocol (human checkpoint)

Every additional variant raises the chance of a lucky pass, so **the budget is fixed in advance**.
Run exactly these commands, in this order. Do not add or change runs after seeing results.

Grid for all runs: `--horizons 6,12,18 --tp-atr-mults 3.0,4.0` (6 cells per symbol).

| Run | Command flags (plus the grid and `--holdout-start 2025-10-01`) |
|---|---|
| R2-A | *(none)* — Round 1 features with the new metrics |
| R2-B | `--pooled` |
| R2-C | `--pooled --feature-groups cross_asset,calendar,derivs_extra` |
| R2-D | `--pooled --feature-groups cross_asset,calendar,derivs_extra --model-size small` |
| R2-W | Same flags as the best of A–D by the rule below, plus `--taker-fee 0.0002` (**what-if only**) |

Example:

```bash
uv run trader-bot barrier-experiment --holdout-start 2025-10-01 \
  --horizons 6,12,18 --tp-atr-mults 3.0,4.0 --pooled \
  --feature-groups cross_asset,calendar,derivs_extra
```

Total: 4 × 6 = 24 selection cells per symbol. Write this number in `ISSUES.md`.

### Selection rule (stricter than Round 1)

A cell passes only if, for **both** symbols:

- `rr_meets_risk_floor` is true,
- `folds_with_signals >= 4`,
- `folds_with_positive_nonoverlap_r >= 4`,
- `n_trades_nonoverlap >= 60`,
- `t_stat_nonoverlap >= 2.5` (about 2.0 plus a margin for 24 cells),
- `mean_net_r_nonoverlap > max(always_long_mean_net_r, always_short_mean_net_r, 0)`.

Among passing cells pick the highest average `mean_net_r_nonoverlap`. R2-W never counts as a
pass; it only answers "would cheaper execution change the picture?".

### Outcomes

- **No cell passes (the likely outcome):** append the R2-A…R2-W summary tables to `ISSUES.md` with
  the conclusion "no robust 4h directional edge after costs with available data". Do not run the
  holdout. Stop and hand off to a human, who decides on Round 3 (see below).
- **A cell passes:** write its settings into `config/strategy_barrier.yaml`, a copy of
  `config/strategy.yaml` with `signal_mode: barrier`, the chosen `horizon_candles`,
  `barrier.tp_atr_mult`, `barrier.model_size` and feature groups. Then:
  - if the winner is from R2-A, run **once**:
    `STRATEGY_CONFIG=config/strategy_barrier.yaml uv run trader-bot barrier-backtest --holdout-start 2025-10-01`
  - if the winner is pooled (R2-B–D), `barrier-backtest` does not support pooled models yet.
    Stop and ask a human to approve an extra phase that adds pooled walk-forward predictions
    (same pattern as `walk_forward_barrier_predictions` on the pooled dataset, with the k-scaled
    purge from `cv_evaluate_pooled`), to be done before Phase 19.

  Record validation and holdout metrics, `sharpe_ci_90` and the baseline comparison in `ISSUES.md`.
  Phase 19 (live wiring) still requires explicit human approval and a holdout gate pass.

---

## Phase 19 — Live wiring (only after human approval)

Prerequisite: holdout segment passes the gate (`gate_passed: true`), the lower bound of
`sharpe_ci_90` is > 0, and the barrier result beats every baseline on the holdout.

1. `models/registry.py`: add `should_promote_barrier(symbol, metrics, min_signals, min_mean_r)`:
   return `False` unless `metrics["n_signals"] >= min_signals` and
   `metrics["mean_net_r_on_signals"] >= min_mean_r` (finite). If a current model exists **and**
   its meta has `signal_mode == "barrier"`, also require
   `new.mean_net_r_on_signals * new.signal_coverage > old.mean_net_r_on_signals * old.signal_coverage`.
   Add unit tests mirroring `tests/test_model_selection.py`.
2. `models/retrain.py`: add `retrain_barrier_symbol(engine, cfg, registry, symbol,
   promote_model=True)` that builds `build_barrier_dataset`, runs `cv_evaluate_barrier`, fits a
   final `BarrierModel` on all labeled rows, saves with extra
   `{"signal_mode": "barrier", "horizon_candles": ..., "tp_atr_mult": ..., "atr_stop_mult": ...}`,
   and promotes via `should_promote_barrier`. In `retrain_symbol`, at the very top, if
   `cfg.signal_mode == "barrier"` return `retrain_barrier_symbol(...)`.
3. `scheduler/runner.py` `decide_and_trade`: after loading the model, read
   `registry.meta(symbol, version).get("signal_mode", "classic")`; if it differs from
   `cfg.signal_mode`, log `decision.model_mode_mismatch` and return `False`. If barrier mode:
   ```python
   pred = ens.predict(row).iloc[0]
   price, atr = float(row["ctx_close"].iloc[0]), float(row["ctx_atr"].iloc[0])
   sig = make_ev_signal(pred["p_long_win"], pred["p_short_win"], price, atr,
                        cfg.risk.atr_stop_mult, cfg.barrier.tp_atr_mult,
                        round_trip_cost(cfg.backtest.taker_fee, cfg.backtest.slippage_bps),
                        cfg.barrier.min_ev_r)
   ```
   and continue with the existing `plan_trade` / `open_trade` / `_log_signal` code unchanged.
4. Add a test for the mode-mismatch guard using a fake registry object.
5. Testnet only. Run `uv run trader-bot train` then `uv run trader-bot run --testnet` with the
   barrier config, observe at least 2 weeks of decisions in the `signals` table before any
   further step.

Commit: `Wire barrier mode into retrain and live loop (testnet)`

---

## Round 3 options (human decision; do not start automatically)

Only if Round 2 finds no edge. Each changes the research question, so each needs a fresh holdout
date that no earlier run has evaluated.

| Option | Idea | Rough cost |
|---|---|---|
| Lower-cost execution | Simulate maker-limit entries (fill only if the next bar trades through the limit price), then evaluate. Execution changes need `docs/risk-and-execution.md` review. | Medium |
| Longer timeframe | Daily decisions (`decision_interval: 1d`); costs become small relative to ATR. Needs builder/label/backtest grid generalisation. | Medium–high |
| Regime filter only | Use the model only to turn a simple rule (e.g. `breakout_20`) on or off, instead of predicting direction. | Low |
| Stop here | Treat the strategy as unproven and keep the repository as research tooling. | None |

---

## Backlog (optional, each is a separate small task; same rules apply)

Implement only after Phase 18, one at a time, each evaluated with `barrier-experiment` on
pre-holdout data. Never re-use the holdout for selection. Former items D2–D4 and E1–E3 are now
Phases 14–16; H1 is done (history already starts in 2019-09).

| ID | Change | Where | Acceptance |
|---|---|---|---|
| D1 | **Label-uniqueness sample weights**: weight_i = mean over bars in [i+1, i+h] of 1/(number of labels covering that bar); pass `sample_weight` to both `fit` calls in `BarrierModel`. | `models/barrier.py` | Unit test: non-overlapping labels → all weights 1.0; fully overlapping → 1/h in the middle. |
| G1 | **Nested CV for `tune`**: tune only on the training part of each outer fold. | `models/train.py` | Reported CV metrics come only from outer test folds. |

D1 note: with a fixed horizon every label covers the same number of bars, so weights are
nearly constant. Only worth doing if weights use the actual barrier-exit bar instead of `h`.

---

## Definition of done (Round 2, Phases 12–17)

- [ ] All tests pass; the test count grew.
- [ ] `uv run ruff check src tests` reports no **new** errors compared with before Phase 12.
- [ ] `artifacts/` is untracked and ignored; the local files still exist.
- [ ] Classic `backtest` and the Round 1 `barrier-experiment` defaults produce unchanged results
      (all new flags default to off/default).
- [ ] `ISSUES.md` contains the Round 1 summary.
