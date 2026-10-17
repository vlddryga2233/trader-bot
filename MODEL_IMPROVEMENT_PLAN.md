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
  holdout. Stop and hand off to a human, who decides on Round 3.
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

# Round 3 — regime filter, daily timeframe, limit-order entry

## Why Round 3

Round 2 found no robust 4h directional edge (see `ISSUES.md`). Round 3 tests three different
research questions, chosen by a human on 2026-10-17:

| Direction | Question | Phase |
|---|---|---|
| Regime filter | Can the model improve a simple rule by skipping its bad trades (meta-labeling)? | 21 |
| Daily timeframe | Is there an edge on 1d decisions, where costs are about 0.02 R instead of 0.07 R? | 22 |
| Limit-order entry | Do the existing 4h signals become profitable with a limit entry below/above the market? | 23 |

Expectations, stated before any run:

- **Fees alone are not the lever.** R2-W showed a lower fee adds only about 0.03 R. A limit
  entry helps only through a better entry price, which is offset by missed trades and by fills
  that happen mostly when price moves against the order.
- **Daily data is small:** about 2,200 rows per symbol before the holdout, so at most ~200
  non-overlapping trades per symbol in CV and ~35 in the holdout. Statistical power is low.
- **Live exits stay as they are:** `execution/manager.py` places `STOP_MARKET` and
  `TAKE_PROFIT_MARKET`, so exits pay taker fees and slippage in every simulation. Only the entry
  changes in Phase 23, and only in research code.

Same ground rules as Rounds 1–2: new behaviour is opt-in, defaults do not change, `risk/limits.py`,
`risk/kill_switch.py` and `execution/` are not touched, tests use synthetic data only.

---

## Phase 20 — Shared preparation

### 20.1 Make the Round 2 record unambiguous

`ISSUES.md`: the R2-A and R2-B headings print identical settings because the pooled flag is not
shown. Rename the headings:

- `### R2-B — {` → `### R2-B (pooled) — {`
- `### R2-C — {` → `### R2-C (pooled) — {`
- `### R2-D — {` → `### R2-D (pooled) — {`

### 20.2 Allow a single-side `BarrierModel`

File: `src/trader_bot/models/barrier.py`, class `BarrierModel`.

1. Change the `fit` signature to
   `def fit(self, X, y_long: pd.Series, y_short: pd.Series | None) -> BarrierModel:`
   and make the first line inside the `for side, y in ...` loop:
   ```python
            if y is None:
                continue
   ```
2. In `predict`, replace `for side in ("long", "short"):` with `for side in self.models:` and
   replace the return line with:
   ```python
        return out[[col for col in BARRIER_PRED_COLS if col in out]]
   ```

Existing callers pass both sides, so their behaviour is unchanged.

Test — add to `tests/test_barrier_model.py`:

```python
def test_barrier_model_can_fit_one_side():
    ds = _synthetic()
    model = BarrierModel(ds.columns, purge=12, calib_frac=0.3).fit(ds.X, ds.y_long, None)
    pred = model.predict(ds.X)
    assert "p_long_win" in pred.columns
    assert "p_short_win" not in pred.columns
```

### 20.3 One selection helper for every experiment

Round 2 gates were counted by hand. Create `src/trader_bot/models/selection.py`
(no ML imports, so it is cheap to test):

```python
"""Selection gates shared by research experiments. A cell passes only if all gates hold."""
from __future__ import annotations


def symbol_gates(m: dict, mode: str, t_min: float, min_trades: int) -> dict[str, bool]:
    """mode: "barrier" (vs naive exposure), "meta" (vs the unfiltered rule), "limit" (vs market)."""
    if mode == "limit":
        mean, t = m.get("limit_mean_net_r_nonoverlap"), m.get("limit_t_stat_nonoverlap")
        positive_folds = m.get("folds_with_positive_limit_r", 0)
        comparator = m.get("mean_net_r_nonoverlap") or 0.0
    else:
        mean, t = m.get("mean_net_r_nonoverlap"), m.get("t_stat_nonoverlap")
        positive_folds = m.get("folds_with_positive_nonoverlap_r", 0)
        if mode == "meta":
            comparator = m.get("primary_mean_net_r_nonoverlap") or 0.0
        else:
            comparator = max(m["always_long_mean_net_r"], m["always_short_mean_net_r"])
    return {
        "rr_floor": bool(m["rr_meets_risk_floor"]),
        "signals_in_4_folds": m.get("folds_with_signals", 0) >= 4,
        "positive_in_4_folds": positive_folds >= 4,
        "enough_trades": m.get("n_trades_nonoverlap", 0) >= min_trades,
        "t_stat": t is not None and t >= t_min,
        "beats_comparator": mean is not None and mean > max(comparator, 0.0),
    }


def select(report: dict, mode: str, t_min: float = 3.0, min_trades: int = 40) -> list[dict]:
    """One row per grid cell, evaluated jointly over all symbols in the report."""
    symbols = list(report["symbols"])
    rows = []
    for cells in zip(*(report["symbols"][s] for s in symbols), strict=True):
        gates = {s: symbol_gates(c["metrics"], mode, t_min, min_trades)
                 for s, c in zip(symbols, cells, strict=True)}
        row = {key: value for key, value in cells[0].items() if key != "metrics"}
        row["gates_met"] = sum(sum(g.values()) for g in gates.values())
        row["gates_total"] = sum(len(g) for g in gates.values())
        row["passed"] = row["gates_met"] == row["gates_total"]
        row["failed"] = {s: [name for name, ok in g.items() if not ok] for s, g in gates.items()}
        rows.append(row)
    return rows
```

Wire it into `barrier-experiment`:

- `models/barrier_experiment.py`: add parameters `t_min: float = 3.0, min_trades: int = 40` to
  `run_barrier_experiment`. Just before `out_dir.mkdir(...)` add
  ```python
    mode = "limit" if cfg.barrier.limit_offset_atr > 0 else "barrier"
    report["selection"] = select(report, mode, t_min, min_trades)
  ```
  (`limit_offset_atr` is added in Phase 23. Until then use `mode = "barrier"` and change the line
  in Phase 23.)
- `cli.py`, `barrier_experiment`: add options
  `t_min: float = typer.Option(3.0, help="Minimum non-overlapping t-statistic")` and
  `min_trades: int = typer.Option(40, help="Minimum non-overlapping trades per symbol")`,
  pass them to `run_barrier_experiment`, and add `"selection": report["selection"]` to the dict
  printed by `typer.echo`.

Create `tests/test_selection.py`:

```python
from trader_bot.models.selection import select, symbol_gates


def _m(**over):
    m = {"rr_meets_risk_floor": True, "folds_with_signals": 5,
         "folds_with_positive_nonoverlap_r": 4, "n_trades_nonoverlap": 80,
         "t_stat_nonoverlap": 3.5, "mean_net_r_nonoverlap": 0.2,
         "always_long_mean_net_r": -0.05, "always_short_mean_net_r": -0.06}
    m.update(over)
    return m


def test_all_gates_pass():
    assert all(symbol_gates(_m(), "barrier", 3.0, 40).values())


def test_each_gate_can_fail():
    assert not symbol_gates(_m(t_stat_nonoverlap=2.9), "barrier", 3.0, 40)["t_stat"]
    assert not symbol_gates(_m(n_trades_nonoverlap=39), "barrier", 3.0, 40)["enough_trades"]
    assert not symbol_gates(_m(mean_net_r_nonoverlap=-0.01), "barrier", 3.0, 40)[
        "beats_comparator"]


def test_meta_mode_compares_with_unfiltered_rule():
    gates = symbol_gates(_m(primary_mean_net_r_nonoverlap=0.25), "meta", 3.0, 40)
    assert not gates["beats_comparator"]


def test_select_requires_every_symbol():
    report = {"symbols": {
        "BTCUSDT": [{"horizon_candles": 6, "tp_atr_mult": 3.0, "metrics": _m()}],
        "ETHUSDT": [{"horizon_candles": 6, "tp_atr_mult": 3.0, "metrics": _m(folds_with_signals=3)}],
    }}
    row = select(report, "barrier")[0]
    assert not row["passed"]
    assert row["failed"]["ETHUSDT"] == ["signals_in_4_folds"]
    assert row["gates_met"] == 11
```

### 20.4 Simulator bar length as a parameter

File: `src/trader_bot/backtest/engine.py`, function `simulate`: add a last parameter
`bar: pd.Timedelta = BAR` and replace every `BAR` inside the function body with `bar`
(funding window, `close_t`, `equity_points`).

File: `src/trader_bot/backtest/segments.py`, `simulate_segment`: add a last parameter
`bar: pd.Timedelta = BAR`, use `seg_end - cfg.labeling.horizon_candles * bar` for
`tradable_end`, and pass `bar=bar` to `simulate`.

Test — add to the existing backtest engine tests (the file that already tests `simulate`):
run a one-trade simulation with daily bars, `bar=pd.Timedelta(days=1)` and `horizon=1`, and
assert the trade's `closed_at` equals the entry bar's open time plus one day. Copy the existing
test's setup and change only the bar frequency and the `bar` argument.

Verify: all existing tests still pass (defaults unchanged).

Commit: `Add selection helper, single-side barrier model and simulator bar length`

---

## Phase 21 — Regime filter (meta-labeling)

A simple rule (`breakout_20` or `momentum_42` from `backtest/baselines.py`) chooses the side.
The model is trained only on bars where the rule fires and predicts whether **that** trade hits
the take-profit first. The trade is taken only if its expected value clears `min_ev_r`.
The comparator is the same rule **without** the filter.

### 21.1 Comparator metrics

File: `src/trader_bot/models/barrier.py`, `barrier_signal_metrics`:

1. Add a parameter `primary: np.ndarray | None = None` (direction of the rule per row: +1/-1/0).
2. Change `return { ... }` to `out = { ... }` and then add:

```python
    if primary is not None:
        fire = primary != 0
        primary_r = np.where(primary == 1, rl, rs) - cost_r
        primary_trades = non_overlapping(fire, cfg.labeling.horizon_candles)
        out["primary_n_trades_nonoverlap"] = int(primary_trades.sum())
        out["primary_mean_net_r_nonoverlap"] = (
            float(primary_r[primary_trades].mean()) if primary_trades.any() else None
        )
    return out
```

### 21.2 Meta model and CV

Create `src/trader_bot/models/meta.py`:

```python
"""Meta-labeling: a rule picks the side; a calibrated classifier decides whether to trade."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from trader_bot.backtest.baselines import baseline_direction
from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.models.barrier import (
    FOLD_KEYS, BarrierModel, barrier_signal_metrics, build_barrier_dataset, summarize_folds,
)
from trader_bot.models.cv import purged_walk_forward
from trader_bot.models.selection import select

PRIMARIES = ("breakout_20", "momentum_42")


def _as_barrier_preds(p: np.ndarray, d: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Put P(win of the rule's side) on that side and 0 on the other, so the EV rule and
    barrier_signal_metrics can be reused unchanged (p = 0 never clears min_ev_r)."""
    return pd.DataFrame({"p_long_win": np.where(d == 1, p, 0.0),
                         "p_short_win": np.where(d == -1, p, 0.0)}, index=index)


def cv_evaluate_meta(feats: pd.DataFrame, cfg: StrategyConfig, primary: str) -> dict:
    """Purged CV over all bars; each fold trains only on training bars where the rule fires."""
    ds = build_barrier_dataset(feats, cfg)
    lab = ds.labeled
    X = ds.X.iloc[lab]
    d = baseline_direction(feats, primary).reindex(X.index).fillna(0).astype(int).to_numpy()
    yl, ys = ds.y_long.iloc[lab], ds.y_short.iloc[lab]
    rl, rs = ds.r_long.iloc[lab], ds.r_short.iloc[lab]
    y_rule = pd.Series(np.where(d == 1, yl, ys), index=X.index)
    m, h = cfg.model, cfg.labeling.horizon_candles
    preds, folds = [], []
    for fold, (tr, te) in enumerate(
        purged_walk_forward(len(X), m.cv_splits, h, m.embargo_candles, m.min_train_candles),
        start=1,
    ):
        fire = tr[d[tr] != 0]
        model = BarrierModel(ds.columns, purge=h + m.embargo_candles,
                             calib_frac=cfg.barrier.calib_frac,
                             size=cfg.barrier.model_size).fit(X.iloc[fire], y_rule.iloc[fire], None)
        p = model.predict(X.iloc[te])["p_long_win"].to_numpy()
        P = _as_barrier_preds(p, d[te], X.index[te])
        preds.append(P)
        fm = barrier_signal_metrics(P, X.iloc[te], yl.iloc[te], ys.iloc[te], rl.iloc[te],
                                    rs.iloc[te], cfg, primary=d[te])
        folds.append({"fold": fold, "n_train_rule_rows": len(fire), "n_test": len(te),
                      **{key: fm[key] for key in FOLD_KEYS if key in fm}})
    P = pd.concat(preds)
    pos = X.index.get_indexer(P.index)
    metrics = barrier_signal_metrics(P, X.iloc[pos], yl.iloc[pos], ys.iloc[pos], rl.iloc[pos],
                                     rs.iloc[pos], cfg, primary=d[pos])
    metrics["n_rule_rows"] = int((d != 0).sum())
    return summarize_folds(metrics, folds)


def run_meta_experiment(engine: Engine, cfg: StrategyConfig, out_dir: Path,
                        holdout_start: pd.Timestamp, primaries: list[str], horizons: list[int],
                        tp_atr_mults: list[float], t_min: float = 3.0,
                        min_trades: int = 40) -> dict:
    report: dict = {"holdout_start": str(holdout_start), "mode": "meta", "symbols": {},
                    "settings": {"model_size": cfg.barrier.model_size,
                                 "taker_fee": cfg.backtest.taker_fee,
                                 "slippage_bps": cfg.backtest.slippage_bps}}
    for symbol in cfg.symbols:
        feats, _ = features_for_symbol(engine, symbol, cfg)
        feats = feats.loc[feats.index < holdout_start]
        results = []
        for primary in primaries:
            for horizon in horizons:
                for tp in tp_atr_mults:
                    variant = cfg.model_copy(update={
                        "labeling": cfg.labeling.model_copy(update={"horizon_candles": horizon}),
                        "barrier": cfg.barrier.model_copy(update={"tp_atr_mult": tp}),
                    })
                    results.append({"primary": primary, "horizon_candles": horizon,
                                    "tp_atr_mult": tp,
                                    "metrics": cv_evaluate_meta(feats, variant, primary)})
        report["symbols"][symbol] = results
    report["selection"] = select(report, "meta", t_min, min_trades)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
```

Notes:
- The rule direction is computed on the full feature frame with trailing data only
  (`baseline_direction` uses `.shift(1)` / trailing windows), then aligned to labeled rows.
- In `models/pooled.py`, change `**{key: fm[key] for key in FOLD_KEYS}` to
  `**{key: fm[key] for key in FOLD_KEYS if key in fm}` (needed after Phase 23 adds an optional
  fold key; harmless now).

### 21.3 CLI

File: `src/trader_bot/cli.py`. Add a command after `barrier_experiment`, using the same parsing
and validation style:

```python
@app.command("meta-experiment")
def meta_experiment(
    holdout_start: str = typer.Option("2025-10-01", help="Rows from this date are excluded"),
    primaries: str = typer.Option("breakout_20,momentum_42", help="Comma-separated rules"),
    horizons: str = typer.Option("6,12", help="Comma-separated horizons in candles"),
    tp_atr_mults: str = typer.Option("3.0", help="Comma-separated take-profit ATR mults"),
    t_min: float = typer.Option(3.0, help="Minimum non-overlapping t-statistic"),
    min_trades: int = typer.Option(40, help="Minimum non-overlapping trades per symbol"),
) -> None:
    """Purged-CV grid for a rule + model filter (meta-labeling), pre-holdout data only."""
    import pandas as pd

    from trader_bot.models.meta import PRIMARIES, run_meta_experiment

    _, cfg, engine = _ctx()
    rules = [value.strip() for value in primaries.split(",") if value.strip()]
    if not rules or not set(rules) <= set(PRIMARIES):
        raise typer.BadParameter(f"primaries must be in {list(PRIMARIES)}")
    try:
        tps = [float(value) for value in tp_atr_mults.split(",")]
        hs = [int(value) for value in horizons.split(",")]
    except ValueError as exc:
        raise typer.BadParameter("use comma-separated numbers") from exc
    if any(value <= 0 for value in tps) or any(value <= 0 for value in hs):
        raise typer.BadParameter("values must be positive")
    holdout = pd.Timestamp(_parse_utc(holdout_start, "holdout_start"))
    out = Path("artifacts/experiments") / datetime.now(UTC).strftime("meta-%Y%m%dT%H%M%SZ")
    report = run_meta_experiment(engine, cfg, out, holdout, rules, hs, tps, t_min, min_trades)
    typer.echo(json.dumps({"report": str(out / "report.json"),
                           "selection": report["selection"]}, indent=2, default=str))
```

### 21.4 Tests

Add to `tests/test_barrier_model.py`:

```python
def _feats(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC", name="decision_time")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    f = pd.DataFrame({"ctx_close": close, "ctx_high": close * 1.005, "ctx_low": close * 0.995,
                      "ctx_atr": close * 0.01}, index=idx)
    f["ret_42"] = np.log(f["ctx_close"]).diff(42)
    f["vol_42"] = np.log(f["ctx_close"]).diff().rolling(42).std()
    f["atr_pct"] = 0.01
    f["f1"] = rng.normal(size=n)
    return f


def test_signal_metrics_report_unfiltered_rule():
    ds = _synthetic()
    P = pd.DataFrame({"p_long_win": 0.9, "p_short_win": 0.0}, index=ds.X.index)
    m = barrier_signal_metrics(P, ds.X, ds.y_long, ds.y_short, ds.r_long, ds.r_short,
                               StrategyConfig(), primary=np.ones(len(P), dtype=int))
    assert m["primary_n_trades_nonoverlap"] > 0
    assert m["primary_mean_net_r_nonoverlap"] is not None


def test_cv_evaluate_meta_runs_and_reports_comparator():
    from trader_bot.models.meta import cv_evaluate_meta

    cfg = StrategyConfig()
    cfg.model = cfg.model.model_copy(update={"min_train_candles": 300, "cv_splits": 2})
    cfg.labeling = cfg.labeling.model_copy(update={"horizon_candles": 12})
    out = cv_evaluate_meta(_feats(), cfg, "momentum_42")
    assert len(out["cv_folds"]) == 2
    assert "primary_mean_net_r_nonoverlap" in out
    assert out["n_rule_rows"] > 0
```

If `barrier_signal_metrics` is not yet imported in this test file, add it to the existing
`from trader_bot.models.barrier import ...` line.

Commit: `Add meta-labeling regime filter experiment`

---

## Phase 22 — Daily timeframe (research only)

### 22.1 Config

File: `src/trader_bot/config/settings.py`:

- `StrategyConfig.decision_interval: Literal["4h", "1d"] = "4h"`.
- In `load_strategy`, before `return cfg`, add a guard so a config file cannot switch the live
  loop to daily (the scheduler and live features are 4h only):
  ```python
    if cfg.decision_interval != "4h":
        raise ValueError("decision_interval 1d is research-only; use --interval 1d")
  ```

### 22.2 Daily features

Create `src/trader_bot/features/daily.py`:

```python
"""Daily decision features (research only). Index = decision time = daily candle close."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader_bot.features.builder import CTX_PREFIX, _asof, _atr, _stamp_closed

DAY = pd.Timedelta(days=1)


def daily_features(kd: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    c = kd["close"]
    lc = np.log(c)
    r1 = lc.diff()
    p = pd.DataFrame(index=kd.index)
    for n in (1, 3, 7, 14, 42):
        p[f"ret_{n}"] = lc.diff(n)
    p["vol_7"] = r1.rolling(7).std()
    p["vol_42"] = r1.rolling(42).std()
    atr = _atr(kd)
    p["atr_pct"] = atr / c
    p["range_pct"] = (kd["high"] - kd["low"]) / c
    p["close_pos"] = (c - kd["low"]) / (kd["high"] - kd["low"]).replace(0, np.nan)
    v = kd["volume"]
    p["vol_z_30"] = (v - v.rolling(30).mean()) / v.rolling(30).std()
    p["taker_buy_ratio"] = kd["taker_buy_base"] / v.replace(0, np.nan)
    p["dist_sma_20"] = c / c.rolling(20).mean() - 1
    p["dist_sma_50"] = c / c.rolling(50).mean() - 1
    p[f"{CTX_PREFIX}close"] = c
    p[f"{CTX_PREFIX}atr"] = atr
    p[f"{CTX_PREFIX}high"] = kd["high"]
    p[f"{CTX_PREFIX}low"] = kd["low"]
    p = _stamp_closed(p, DAY)
    if not funding.empty:
        rate = funding["rate"]
        f = pd.DataFrame({
            "funding_mean_3": rate.rolling(3).mean(),
            "funding_mean_21": rate.rolling(21).mean(),
            "funding_z_90": (rate - rate.rolling(90).mean()) / rate.rolling(90).std(),
        })
        p = _asof(p, f)
    p.index.name = "decision_time"
    return p
```

`ret_42`, `vol_42` and `atr_pct` are deliberately named like the 4h columns: they are the
`CORE_FEATURES` that `build_barrier_dataset` requires (here they mean 42 **days**).

File: `src/trader_bot/features/pipeline.py`, `features_for_symbol`: right after
`raw = load_raw(engine, symbol)` add

```python
    if cfg.decision_interval == "1d":
        from trader_bot.features.daily import daily_features

        return daily_features(raw["kd"], raw["funding"]), raw
```

### 22.3 Labels use the configured step

File: `src/trader_bot/models/barrier.py`, `build_barrier_dataset`: pass
`step=pd.Timedelta(cfg.decision_interval)` to `make_barrier_labels`. (`pd.Timedelta("4h")`
equals the current default, so 4h results do not change.)

### 22.4 CLI

File: `src/trader_bot/cli.py`, `barrier_experiment`: add options

```python
    interval: str = typer.Option("4h", help="4h or 1d (1d is research-only)"),
    min_train_candles: int | None = typer.Option(None, help="Override model.min_train_candles"),
```

and apply them before running:

```python
    if interval not in ("4h", "1d"):
        raise typer.BadParameter("interval must be 4h or 1d")
    cfg.decision_interval = interval
    if min_train_candles is not None:
        cfg.model = cfg.model.model_copy(update={"min_train_candles": min_train_candles})
```

Add `"decision_interval": cfg.decision_interval` and
`"min_train_candles": cfg.model.min_train_candles` to `report["settings"]` in
`run_barrier_experiment`.

Note: with `--interval 1d`, `horizons` and `embargo_candles` are counted in **days**
(`embargo_candles: 6` = 6 days). The default `min_train_candles: 1500` is too large for ~2,200
daily rows; the protocol uses `--min-train-candles 730`. `--feature-groups` has no effect on
daily features.

### 22.5 Tests

Create `tests/test_daily_features.py`:

```python
import numpy as np
import pandas as pd

from trader_bot.features.daily import daily_features
from trader_bot.models.train import CORE_FEATURES


def _kd(n=300, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC", name="open_time")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    df = pd.DataFrame({"close": close}, index=idx)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * 1.01
    df["low"] = df[["open", "close"]].min(axis=1) * 0.99
    df["volume"] = rng.uniform(1, 10, n)
    df["taker_buy_base"] = df["volume"] * rng.uniform(0.4, 0.6, n)
    return df


def _funding(n=900, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="8h", tz="UTC", name="ts")
    return pd.DataFrame({"rate": rng.normal(1e-4, 1e-4, n)}, index=idx)


def test_daily_features_have_core_columns_and_close_time_index():
    kd = _kd()
    feats = daily_features(kd, _funding())
    assert set(CORE_FEATURES) <= set(feats.columns)
    assert feats.index[0] == kd.index[0] + pd.Timedelta(days=1)


def test_daily_features_do_not_leak_future():
    kd, funding = _kd(), _funding()
    full = daily_features(kd, funding)
    t = full.index[200]
    cut = daily_features(kd[kd.index + pd.Timedelta(days=1) <= t], funding[funding.index <= t])
    pd.testing.assert_series_equal(full.loc[t], cut.loc[t], check_names=False)
```

Commit: `Add research-only daily decision interval for barrier experiments`

---

## Phase 23 — Limit-order entry what-if (label space)

Same 4h signals as Round 2; only the entry changes. At the decision close, a limit order is
placed `limit_offset_atr × ATR` better than the close (below for long, above for short), valid
for one candle. It fills only if the next candle trades **through** the price (strict `<` / `>`).
Unfilled signals count as 0 R (no trade, no cost). On the fill candle only the stop is checked;
the target is checked from the next candle (conservative, because the order of prices inside
the candle is unknown). Costs: maker fee on entry, taker fee and slippage on exit.

### 23.1 Config

File: `src/trader_bot/config/settings.py`:

- `BacktestCfg.maker_fee: float = 0.0002`
- `BarrierCfg.limit_offset_atr: float = Field(default=0.0, ge=0, le=1)` (0 = market entry, current
  behaviour)

### 23.2 Outcomes

File: `src/trader_bot/labeling/barriers.py`, add below `make_barrier_labels`:

```python
def limit_entry_outcomes(feats: pd.DataFrame, horizon: int, stop_mult: float, tp_mult: float,
                         offset_atr: float) -> pd.DataFrame:
    """R outcome of a one-candle limit entry offset_atr*ATR better than the decision close.

    Unfilled rows: r = 0, filled = False. R is measured against stop_mult*ATR from the fill.
    """
    close = feats["ctx_close"].to_numpy(dtype=float)
    high = feats["ctx_high"].to_numpy(dtype=float)
    low = feats["ctx_low"].to_numpy(dtype=float)
    atr = feats["ctx_atr"].to_numpy(dtype=float)
    n = len(feats)
    stop_d = stop_mult * atr
    tp_d = tp_mult * atr
    rr = tp_mult / stop_mult
    out = pd.DataFrame(index=feats.index)
    for side, name in ((1, "long"), (-1, "short")):
        entry = close - side * offset_atr * atr
        r = np.full(n, np.nan)
        filled = np.zeros(n, dtype=bool)
        for i in range(max(n - horizon, 0)):
            if not (np.isfinite(stop_d[i]) and stop_d[i] > 0):
                continue
            touched = low[i + 1] < entry[i] if side == 1 else high[i + 1] > entry[i]
            if not touched:
                r[i] = 0.0
                continue
            filled[i] = True
            stop = entry[i] - side * stop_d[i]
            tp = entry[i] + side * tp_d[i]
            result = None
            for j in range(1, horizon + 1):
                if (low[i + j] <= stop) if side == 1 else (high[i + j] >= stop):
                    result = -1.0
                    break
                if j > 1 and ((high[i + j] >= tp) if side == 1 else (low[i + j] <= tp)):
                    result = rr
                    break
            if result is None:
                result = side * (close[i + horizon] - entry[i]) / stop_d[i]
            r[i] = result
        out[f"r_{name}_limit"] = r
        out[f"filled_{name}"] = filled
    return out
```

### 23.3 Dataset and metrics

File: `src/trader_bot/models/barrier.py`:

1. `BarrierDataset`: add a last field `limit: pd.DataFrame | None = None`.
2. `build_barrier_dataset`: before `return`, add
   ```python
    limit = None
    if cfg.barrier.limit_offset_atr > 0:
        limit = limit_entry_outcomes(valid, cfg.labeling.horizon_candles,
                                     cfg.risk.atr_stop_mult, cfg.barrier.tp_atr_mult,
                                     cfg.barrier.limit_offset_atr)
   ```
   and pass `limit` as the last argument to `BarrierDataset(...)`. Import
   `limit_entry_outcomes` next to `make_barrier_labels`.
3. `barrier_signal_metrics`: add a parameter `limit: pd.DataFrame | None = None` and, before
   `return out`, add:
   ```python
    if limit is not None:
        is_long = direction == 1
        filled = np.where(is_long, limit["filled_long"].to_numpy(),
                          limit["filled_short"].to_numpy()).astype(bool)
        bt = cfg.backtest
        limit_cost_r = (bt.maker_fee + bt.taker_fee + bt.slippage_bps / 10_000) * price / (
            stop_mult * atr
        )
        limit_r = np.where(is_long, limit["r_long_limit"].to_numpy(),
                           limit["r_short_limit"].to_numpy()) - np.where(filled, limit_cost_r, 0.0)
        limit_trade_r = limit_r[trades]
        out["limit_fill_rate"] = float(filled[trades].mean()) if trades.any() else None
        out["limit_mean_net_r_nonoverlap"] = (
            float(limit_trade_r.mean()) if len(limit_trade_r) else None
        )
        out["limit_t_stat_nonoverlap"] = _t_stat(limit_trade_r)
   ```
4. `FOLD_KEYS`: append `"limit_mean_net_r_nonoverlap"`. In `cv_evaluate_barrier`, change the
   fold dict to `**{key: fold_metrics[key] for key in FOLD_KEYS if key in fold_metrics}`.
5. `cv_evaluate_barrier`: add `lim = ds.limit.iloc[lab] if ds.limit is not None else None`
   after the `rl, rs = ...` line; pass `limit=lim.iloc[te] if lim is not None else None` to the
   per-fold `barrier_signal_metrics` call and `limit=lim.loc[idx] if lim is not None else None`
   to the final call.
6. `summarize_folds`: before `return metrics` add
   ```python
    if folds and "limit_mean_net_r_nonoverlap" in folds[0]:
        metrics["folds_with_positive_limit_r"] = positive("limit_mean_net_r_nonoverlap")
   ```

The signals and their EV are unchanged (still priced with market costs); only the realised
outcome is recomputed for the limit entry.

### 23.4 CLI and report

- `cli.py`, `barrier_experiment`: add
  `limit_offset_atr: float = typer.Option(0.0, help="Limit entry offset in ATR (0 = market)")`.
  Reject values outside `[0, 1]`, and reject `--pooled` together with a non-zero offset
  (`typer.BadParameter("limit entry is not supported with --pooled")`). Apply with
  `cfg.barrier = cfg.barrier.model_copy(update={"limit_offset_atr": limit_offset_atr})`.
- `models/barrier_experiment.py`: add `"limit_offset_atr"` and `"maker_fee"` to
  `report["settings"]`; set `mode = "limit" if cfg.barrier.limit_offset_atr > 0 else "barrier"`
  (see 20.3).

### 23.5 Tests

Add to `tests/test_barriers.py`:

```python
from trader_bot.labeling.barriers import limit_entry_outcomes


def _path(highs, lows):
    idx = pd.date_range("2024-01-01", periods=len(highs), freq="4h", tz="UTC")
    return pd.DataFrame({"ctx_close": 100.0, "ctx_atr": 1.0, "ctx_high": highs,
                         "ctx_low": lows}, index=idx)


def test_limit_entry_not_filled_without_trading_through():
    out = limit_entry_outcomes(_path([100, 100.5, 100.5], [100, 99.6, 99.6]), 2, 2.0, 4.0, 0.5)
    assert out["r_long_limit"].iloc[0] == 0.0 and not out["filled_long"].iloc[0]
    assert out["r_short_limit"].iloc[0] == 0.0 and not out["filled_short"].iloc[0]
    assert np.isnan(out["r_long_limit"].iloc[1])


def test_limit_entry_fill_then_target():
    out = limit_entry_outcomes(_path([100, 100, 104], [100, 99.4, 99.4]), 2, 2.0, 4.0, 0.5)
    assert out["filled_long"].iloc[0]
    assert out["r_long_limit"].iloc[0] == 2.0


def test_limit_entry_stop_on_fill_candle():
    out = limit_entry_outcomes(_path([100, 100, 100], [100, 97.0, 99.0]), 2, 2.0, 4.0, 0.5)
    assert out["r_long_limit"].iloc[0] == -1.0
```

(Entry 99.5, stop 97.5, target 103.5 in these cases. Add `import numpy as np` / `import pandas as pd`
at the top of the file if they are not already imported.)

Commit: `Add limit-entry what-if to barrier experiments`

---

## Phase 24 — Round 3 research protocol (human checkpoint)

Same pre-holdout data as before (`--holdout-start 2025-10-01`). The budget is fixed in advance:
**run exactly these commands, once, in this order**, and do not add runs after seeing results.

| Run | Command | Cells per symbol |
|---|---|---|
| R3-M | `uv run trader-bot meta-experiment --holdout-start 2025-10-01 --primaries breakout_20,momentum_42 --horizons 6,12 --tp-atr-mults 3.0` | 4 |
| R3-D | `uv run trader-bot barrier-experiment --holdout-start 2025-10-01 --interval 1d --min-train-candles 730 --pooled --horizons 5,10 --tp-atr-mults 3.0,4.0` | 4 |
| R3-L1 | `uv run trader-bot barrier-experiment --holdout-start 2025-10-01 --horizons 12,18 --tp-atr-mults 4.0 --limit-offset-atr 0.25` | 2 |
| R3-L2 | same as R3-L1 with `--limit-offset-atr 0.5` | 2 |

Total: 12 cells per symbol. Every command uses the defaults `--t-min 3.0 --min-trades 40`.

### Selection rule

Read `report["selection"]`; do not recount by hand. A cell passes only if `passed` is true,
meaning for **both** symbols:

- `rr_meets_risk_floor`,
- signals in ≥ 4 of 5 folds and positive non-overlapping mean R in ≥ 4 of 5 folds,
- ≥ 40 non-overlapping trades,
- t-statistic ≥ 3.0 (stricter than Round 2 because many cells have now been tried in total),
- mean non-overlapping R above the comparator and above 0. The comparator is naive exposure
  for R3-D, the **unfiltered rule** for R3-M, and the **market entry of the same cell** for
  R3-L1/L2.

If several cells pass, pick the one with the highest average non-overlapping mean R
(for R3-L cells, `limit_mean_net_r_nonoverlap`).

### Record

Append `## Round 3 (date)` to `ISSUES.md` with, for each run, a table of
`horizon, tp, (primary), gates_met/gates_total, passed`, plus BTC and ETH non-overlapping
mean R, t-statistic and trade count, and, for R3-L, `limit_fill_rate`. Write the conclusion in
one sentence.

### Outcomes

- **Nothing passes:** write "no edge found in Rounds 1–3 with available data; strategy unproven"
  in `ISSUES.md` and **stop**. Do not run any holdout. The repository stays research tooling.
- **A cell passes:** do Phase 25 for that direction only, then run the holdout **once** for the
  single best cell. Record the results and stop for human review. Because the baselines were
  already evaluated on this holdout (Round 1), an R3-M pass must also beat the unfiltered rule
  on the holdout to count. Even with a holdout pass, Phase 19 (live wiring) needs explicit
  human approval and a forward period of at least three months on testnet first.

---

## Phase 25 — Holdout runner for the winning direction (only if Phase 24 has a pass)

Implement **only** the sub-section for the direction that passed. File:
`src/trader_bot/backtest/run_barrier.py`, `run_barrier_backtest`, and the `barrier-backtest`
CLI command.

### 25.1 Daily winner

- CLI: add `--interval` (`4h`/`1d`) and `--min-train-candles` exactly as in 22.4.
- `run_barrier_backtest`: `bar = pd.Timedelta(cfg.decision_interval)`; use
  `raw["kd"] if cfg.decision_interval == "1d" else raw["k4"]` as `"bars"`; use `+ bar` instead of
  `+ BAR` for `end`; pass `bar=bar` to `simulate_segment`.
- The config copy (`config/strategy_barrier.yaml`) keeps `decision_interval: "4h"` because of the
  guard in 22.1; the interval comes only from `--interval 1d`.
- The pooled model is not supported by `walk_forward_barrier_predictions`. If the passing R3-D
  cell needs pooling, stop and ask a human (same rule as Phase 18).

### 25.2 Regime-filter winner

Add to `models/meta.py`:

```python
def walk_forward_meta_predictions(feats: pd.DataFrame, cfg: StrategyConfig, primary: str,
                                  start: pd.Timestamp) -> pd.DataFrame:
    """Same schedule as walk_forward_barrier_predictions; trains only on rule bars."""
    ds = build_barrier_dataset(feats, cfg)
    d = baseline_direction(feats, primary).reindex(ds.X.index).fillna(0).astype(int).to_numpy()
    y_rule = pd.Series(np.where(d == 1, ds.y_long, ds.y_short), index=ds.X.index)
    m, h = cfg.model, cfg.labeling.horizon_candles
    first = max(int(ds.X.index.searchsorted(start)), m.min_train_candles)
    has = ds.has_label.to_numpy()
    out = []
    for s in range(first, len(ds.X), m.retrain_every_candles):
        e = min(s + m.retrain_every_candles, len(ds.X))
        idx = np.arange(0, max(s - h - m.embargo_candles, 0))
        idx = idx[has[idx] & (d[idx] != 0)]
        if len(idx) < 200:
            continue
        model = BarrierModel(ds.columns, purge=h + m.embargo_candles,
                             calib_frac=cfg.barrier.calib_frac,
                             size=cfg.barrier.model_size).fit(ds.X.iloc[idx], y_rule.iloc[idx], None)
        p = model.predict(ds.X.iloc[s:e])["p_long_win"].to_numpy()
        out.append(_as_barrier_preds(p, d[s:e], ds.X.index[s:e]))
    return pd.concat(out) if out else pd.DataFrame(columns=["p_long_win", "p_short_win"])
```

In `run_barrier_backtest` add a parameter `primary: str | None = None`; when set, use
`walk_forward_meta_predictions(feats, cfg, primary, start)` instead of
`walk_forward_barrier_predictions`. CLI: `--primary` option on `barrier-backtest`
(must be in `PRIMARIES`). The EV signal function is unchanged.

For the comparison required in Phase 24, also run
`uv run trader-bot baselines --holdout-start 2025-10-01` (the unfiltered rule's holdout numbers
already exist from Round 1 and must be identical).

### 25.3 Limit-entry winner

File: `src/trader_bot/backtest/engine.py`, `simulate`: add parameters
`limit_offset_atr: float = 0.0, maker_fee: float = 0.0`. Replace the block that starts at
`if sig is not None:` (signal pass → plan → open position) with:

```python
            if sig is not None:
                diag["signal_pass"] += 1
                entry_fee = fee
                entry = bar["open"] * (1 + sig.direction * slip)
                if limit_offset_atr > 0:
                    entry = bar["open"] - sig.direction * limit_offset_atr * float(atr.loc[t])
                    entry_fee = maker_fee
                    touched = bar["low"] < entry if sig.direction == 1 else bar["high"] > entry
                    if not touched:
                        diag["limit_unfilled"] = diag.get("limit_unfilled", 0) + 1
                        sig = None
            if sig is not None:
                plan = plan_trade(sig, cash, entry, float(atr.loc[t]), risk)
                if isinstance(plan, Rejection):
                    reasons = diag["risk_plan_rejections"]
                    reasons[plan.reason] = reasons.get(plan.reason, 0) + 1
                else:
                    diag["risk_plan_approved"] += 1
                    pos = _Pos(plan.direction, plan.qty, entry, plan.stop, plan.take_profit, t)
                    pos.fees += plan.qty * entry * entry_fee
```

and, after the `hit_tp = ...` line, add:

```python
            if limit_offset_atr > 0 and pos.bars == 1:
                hit_tp = False  # fill candle: target not checked, as in limit_entry_outcomes
```

Pass `limit_offset_atr=cfg.barrier.limit_offset_atr, maker_fee=cfg.backtest.maker_fee` from
`simulate_segment` (add both as keyword arguments there with the same defaults, and pass them
from `run_barrier_backtest`). CLI: `--limit-offset-atr` on `barrier-backtest`.

Test: a single-bar case where the limit is not touched (no trade, `limit_unfilled == 1`) and a
case where it fills (entry equals the limit price, entry fee uses `maker_fee`).

Commit (per sub-section): `Support <daily|meta|limit> winner in barrier-backtest`

---

## Backlog (optional, each is a separate small task; same rules apply)

Implement only after Phase 24, one at a time, each evaluated with `barrier-experiment` on
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

## Definition of done (Round 3, Phases 20–23)

- [ ] All tests pass; new tests for selection, single-side model, simulator bar length,
      meta-labeling, daily features and limit-entry outcomes.
- [ ] `uv run ruff check src tests` reports no **new** errors.
- [ ] With default flags, `barrier-experiment` metrics for a Round 2 cell are identical to the
      Round 2 report (only the new `selection` key is added).
- [ ] `load_strategy` rejects `decision_interval: 1d` in a config file.
- [ ] `meta-experiment` appears in `uv run trader-bot --help`; `barrier-experiment --help` lists
      `--interval`, `--min-train-candles`, `--limit-offset-atr`, `--t-min`, `--min-trades`.
- [ ] No changes to `risk/limits.py`, `risk/kill_switch.py`, `execution/`.
