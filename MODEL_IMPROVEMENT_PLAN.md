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

## Phase 0 — Baseline check

1. `uv sync`
2. `uv run pytest -q` → record the number of passing tests.
3. `uv run ruff check src tests` → must be clean (if not, note pre-existing errors; do not fix them).

No commit.

---

## Phase 1 — Pipeline consistency fixes (small, independent)

### 1.1 Apply slippage to take-profit exits

File: `src/trader_bot/backtest/engine.py`, inside `simulate`, replace:

```python
            elif hit_tp:
                close_pos(pos.tp, close_t, "take_profit")
```

with:

```python
            elif hit_tp:
                close_pos(pos.tp * (1 - pos.direction * slip), close_t, "take_profit")
```

Add test to `tests/test_backtest_exec.py`:

```python
def test_take_profit_exit_pays_slippage():
    bars = _bars([[100, 101, 99.5, 100], [100, 106, 99.5, 105], [105, 105, 104, 105]])
    preds, atr = _long_pred(bars)
    funding = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    bt = BacktestCfg(taker_fee=0.0, slippage_bps=10.0, initial_equity=10_000)
    _, trades = simulate(bars, preds, atr, funding, RISK, bt, 0.75, horizon=6,
                         initial_equity=10_000)
    assert trades.iloc[0]["reason"] == "take_profit"
    assert trades.iloc[0]["exit"] < 105.0
```

### 1.2 Fix misleading config comment

File: `config/strategy.yaml`, change only the comment on `retrain_every_candles`:

```yaml
  retrain_every_candles: 540   # backtest walk-forward retrain cadence (540 x 4h = 90 days); live retrains every 7 days
```

### 1.3 Reuse the promoted model's tuned params when retraining without `--tune`

File: `src/trader_bot/models/registry.py`, add method to `ModelRegistry`:

```python
    def current_params(self, symbol: str) -> dict | None:
        """Hyperparameters saved with the promoted model, if any."""
        version = self.current_version(symbol)
        if version is None:
            return None
        return self.meta(symbol, version).get("params")
```

File: `src/trader_bot/models/retrain.py`, replace:

```python
    params = tune(ds, cfg) if do_tune else None
```

with:

```python
    params = tune(ds, cfg) if do_tune else registry.current_params(symbol)
```

Add test to `tests/test_model_selection.py`:

```python
def test_registry_returns_params_of_promoted_model(tmp_path):
    from trader_bot.models.ensemble import Ensemble

    registry = ModelRegistry(tmp_path)
    assert registry.current_params("BTCUSDT") is None
    version = registry.save("BTCUSDT", Ensemble(["a"]), _metrics(),
                            {"params": {"lgbm": {"num_leaves": 8}}})
    registry.promote("BTCUSDT", version)
    assert registry.current_params("BTCUSDT") == {"lgbm": {"num_leaves": 8}}
```

### 1.4 Let the backtest use the promoted model's params (opt-in)

File: `src/trader_bot/cli.py`, in `backtest(...)`:

1. Add parameter:
   ```python
       registry_params: bool = typer.Option(
           False, "--registry-params", help="Use hyperparameters saved with promoted models"
       ),
   ```
2. Change `_, cfg, engine = _ctx()` to `settings, cfg, engine = _ctx()`.
3. Replace the final line with:
   ```python
    params_by_symbol = None
    if registry_params:
        from trader_bot.models.registry import ModelRegistry

        reg = ModelRegistry(settings.artifacts_dir)
        params_by_symbol = {s: reg.current_params(s) for s in cfg.symbols}
    typer.echo(json.dumps(run_backtest(engine, cfg, out, params_by_symbol), indent=2, default=str))
   ```

File: `src/trader_bot/backtest/run.py`, inside `report["settings"]` add:

```python
            "params_by_symbol": params_by_symbol,
```

Verify: `uv run pytest -q && uv run ruff check src tests` and `uv run trader-bot backtest --help`
shows `--registry-params`.

Commit: `Align backtest and retrain with tuned params; add TP slippage`

---

## Phase 2 — Context columns for barrier labels

Barrier labels need each candle's high and low. Add them as `ctx_` columns (never model inputs).

File: `src/trader_bot/features/builder.py`, in `price_features`, after
`p[f"{CTX_PREFIX}atr"] = atr` add:

```python
    p[f"{CTX_PREFIX}high"] = k4["high"]
    p[f"{CTX_PREFIX}low"] = k4["low"]
```

Add test to `tests/test_features.py` (reuse the file's existing fixture/helper that builds
`full` features; read the file first and follow its pattern):

```python
def test_ctx_high_low_are_not_model_inputs():
    # build features exactly as other tests in this file do, assign to `feats`
    assert "ctx_high" in feats.columns and "ctx_low" in feats.columns
    assert "ctx_high" not in model_columns(feats)
    assert "ctx_low" not in model_columns(feats)
```

Commit: `Expose candle high/low as context columns`

---

## Phase 3 — Triple-barrier labels

Create `src/trader_bot/labeling/barriers.py`:

```python
"""Labels that match the traded outcome: stop, take-profit or time exit, whichever comes first.

Decision at the close of row i. The path is rows i+1..i+horizon (each row's high/low is the
candle that closes at that row's timestamp). The stop is checked before the take-profit within
a candle, matching the backtester. Outcomes are in R units (1R = stop distance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_barrier_labels(
    feats: pd.DataFrame,
    horizon: int,
    stop_mult: float,
    tp_mult: float,
    step: pd.Timedelta = pd.Timedelta(hours=4),
) -> pd.DataFrame:
    """Return r_long, r_short (realized R), y_long, y_short (1.0 if TP hit first) and label_end.

    Rows without a full horizon or without a valid ATR get NaN.
    """
    close = feats["ctx_close"].to_numpy(dtype=float)
    high = feats["ctx_high"].to_numpy(dtype=float)
    low = feats["ctx_low"].to_numpy(dtype=float)
    atr = feats["ctx_atr"].to_numpy(dtype=float)
    n = len(feats)
    stop_d = stop_mult * atr
    tp_d = tp_mult * atr
    rr = tp_mult / stop_mult

    fwd_close = np.full(n, np.nan)
    if n > horizon:
        fwd_close[: n - horizon] = close[horizon:]
    invalid = np.isnan(fwd_close) | ~np.isfinite(stop_d) | (stop_d <= 0)

    out = pd.DataFrame(index=feats.index)
    for side, name in ((1, "long"), (-1, "short")):
        result = np.full(n, np.nan)
        done = np.zeros(n, dtype=bool)
        tp_first = np.zeros(n, dtype=bool)
        for j in range(1, horizon + 1):
            hi = np.full(n, np.nan)
            lo = np.full(n, np.nan)
            if n > j:
                hi[: n - j] = high[j:]
                lo[: n - j] = low[j:]
            if side == 1:
                stop_hit = lo <= close - stop_d
                tp_hit = hi >= close + tp_d
            else:
                stop_hit = hi >= close + stop_d
                tp_hit = lo <= close - tp_d
            new_stop = ~done & stop_hit
            new_tp = ~done & ~stop_hit & tp_hit
            result[new_stop] = -1.0
            result[new_tp] = rr
            tp_first |= new_tp
            done |= new_stop | new_tp
        time_r = side * (fwd_close - close) / stop_d
        result = np.where(done, result, time_r)
        result[invalid] = np.nan
        y = np.where(np.isnan(result), np.nan, tp_first.astype(float))
        out[f"r_{name}"] = result
        out[f"y_{name}"] = y
    out["label_end"] = feats.index + horizon * step
    return out
```

Create `tests/test_barriers.py`:

```python
import numpy as np
import pandas as pd
import pytest

from trader_bot.labeling.barriers import make_barrier_labels


def _feats(high, low, close=None, atr=1.0):
    n = len(high)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "ctx_close": close if close is not None else [100.0] * n,
        "ctx_high": high, "ctx_low": low, "ctx_atr": atr,
    }, index=idx)


def test_take_profit_then_stop_and_time_exit():
    high = [100, 101, 103.5, 100.5, 100.5, 100.5, 100.5, 100.5]
    low = [100, 99.5, 99, 99.5, 99.5, 99.5, 99.5, 99.5]
    lab = make_barrier_labels(_feats(high, low), horizon=3, stop_mult=2.0, tp_mult=3.0)
    assert lab["r_long"].iloc[0] == pytest.approx(1.5)   # TP 103 hit on row 2
    assert lab["y_long"].iloc[0] == 1.0
    assert lab["r_short"].iloc[0] == pytest.approx(-1.0)  # stop 102 hit on row 2
    assert lab["y_short"].iloc[0] == 0.0
    assert lab["r_long"].iloc[3] == pytest.approx(0.0)   # time exit, close unchanged
    assert lab["y_long"].iloc[3] == 0.0
    assert lab[["r_long", "r_short", "y_long", "y_short"]].iloc[-3:].isna().all().all()


def test_stop_wins_when_both_barriers_hit_in_same_candle():
    high = [100, 104, 100, 100, 100, 100]
    low = [100, 97.5, 100, 100, 100, 100]
    lab = make_barrier_labels(_feats(high, low), horizon=3, stop_mult=2.0, tp_mult=3.0)
    assert lab["r_long"].iloc[0] == pytest.approx(-1.0)
    assert lab["r_short"].iloc[0] == pytest.approx(-1.0)


def test_labels_do_not_depend_on_rows_beyond_horizon():
    high = [100, 101, 101, 101, 101, 101, 101, 101]
    low = [100, 99, 99, 99, 99, 99, 99, 99]
    a = make_barrier_labels(_feats(high, low), horizon=3, stop_mult=2.0, tp_mult=3.0)
    high2 = high[:5] + [200, 200, 200]
    b = make_barrier_labels(_feats(high2, low), horizon=3, stop_mult=2.0, tp_mult=3.0)
    assert a["r_long"].iloc[0] == b["r_long"].iloc[0]
    assert a["r_long"].iloc[1] == b["r_long"].iloc[1]


def test_invalid_atr_gives_nan():
    lab = make_barrier_labels(_feats([100] * 6, [100] * 6, atr=np.nan), 3, 2.0, 3.0)
    assert lab["r_long"].isna().all()
```

Commit: `Add triple-barrier labels`

---

## Phase 4 — Configuration (opt-in)

File: `src/trader_bot/config/settings.py`

1. Add `from typing import Literal` to imports.
2. Add after `class RiskCfg`... (place before `BacktestCfg`):

```python
class BarrierCfg(BaseModel):
    tp_atr_mult: float = Field(default=3.0, gt=0)
    min_ev_r: float = Field(default=0.05, ge=0)        # minimum expected value per trade, in R
    calib_frac: float = Field(default=0.2, gt=0, lt=0.5)  # tail of training data for calibration
    promotion_min_signals: int = Field(default=30, ge=1)
    promotion_min_mean_r: float = 0.05
```

3. In `StrategyConfig` add two fields:

```python
    signal_mode: Literal["classic", "barrier"] = "classic"
    barrier: BarrierCfg = BarrierCfg()
```

4. In `load_strategy`, before `return cfg`, add:

```python
    if cfg.signal_mode == "barrier" and (
        cfg.barrier.tp_atr_mult / cfg.risk.atr_stop_mult < cfg.risk.min_reward_risk
    ):
        raise ValueError("barrier.tp_atr_mult / risk.atr_stop_mult is below risk.min_reward_risk")
```

File: `config/strategy.yaml` — append at the end (classic stays active):

```yaml

signal_mode: classic        # "barrier" = triple-barrier labels + expected-value signal (research)
barrier:
  tp_atr_mult: 3.0          # take-profit distance in ATR; must be >= atr_stop_mult * min_reward_risk
  min_ev_r: 0.05            # trade only if expected value after costs >= this many R
  calib_frac: 0.2           # most recent share of each training window used for calibration
  promotion_min_signals: 30
  promotion_min_mean_r: 0.05
```

Add test to `tests/test_risk.py` (or a new `tests/test_settings.py`):

```python
def test_barrier_mode_rejects_tp_below_reward_risk_floor(tmp_path):
    from trader_bot.config.settings import load_strategy

    path = tmp_path / "s.yaml"
    path.write_text("signal_mode: barrier\nbarrier: {tp_atr_mult: 2.0}\n"
                    "risk: {atr_stop_mult: 2.0, min_reward_risk: 1.5}\n")
    with pytest.raises(ValueError):
        load_strategy(path)
```

Commit: `Add opt-in barrier signal mode configuration`

---

## Phase 5 — Expected-value signal rule

File: `src/trader_bot/risk/engine.py` — add (do **not** change `make_signal` or `plan_trade`):

```python
import math


def round_trip_cost(taker_fee: float, slippage_bps: float) -> float:
    """Entry + exit cost as a fraction of price."""
    return 2 * (taker_fee + slippage_bps / 10_000)


def make_ev_signal(p_long_win: float, p_short_win: float, price: float, atr: float,
                   stop_mult: float, tp_mult: float, cost_frac: float,
                   min_ev_r: float) -> Signal | None:
    """Pick the side with the higher expected value in R; trade only if EV >= min_ev_r.

    EV = p * RR - (1 - p) - cost. Time exits are treated as full losses, so EV is conservative.
    The returned pred_ret encodes the fixed take-profit so plan_trade places TP at tp_mult * ATR.
    """
    values = (p_long_win, p_short_win, price, atr)
    if not all(math.isfinite(v) for v in values) or price <= 0 or atr <= 0:
        return None
    stop_dist = max(stop_mult, limits.MIN_ATR_STOP_MULT) * atr
    rr = tp_mult * atr / stop_dist
    cost_r = cost_frac * price / stop_dist
    ev_long = p_long_win * rr - (1 - p_long_win) - cost_r
    ev_short = p_short_win * rr - (1 - p_short_win) - cost_r
    if ev_long >= ev_short:
        direction, prob, ev = 1, p_long_win, ev_long
    else:
        direction, prob, ev = -1, p_short_win, ev_short
    if ev < min_ev_r:
        return None
    # tiny epsilon keeps tp/stop from rounding below an exactly-equal min_reward_risk
    return Signal(direction, prob, direction * tp_mult * atr / price * (1 + 1e-9))
```

Put `import math` at the top of the file with the other imports.

Add tests to `tests/test_risk.py`:

```python
from trader_bot.risk.engine import make_ev_signal, round_trip_cost


def test_ev_signal_long_when_expected_value_positive():
    sig = make_ev_signal(0.5, 0.1, 100.0, 1.0, 2.0, 3.0, 0.0, 0.05)
    assert sig is not None and sig.direction == 1
    assert sig.pred_ret == pytest.approx(0.03)


def test_ev_signal_none_when_expected_value_negative():
    assert make_ev_signal(0.35, 0.1, 100.0, 1.0, 2.0, 3.0, 0.0, 0.05) is None


def test_ev_signal_costs_can_veto():
    assert make_ev_signal(0.44, 0.1, 100.0, 1.0, 2.0, 3.0, 0.0, 0.05) is not None
    assert make_ev_signal(0.44, 0.1, 100.0, 1.0, 2.0, 3.0, 0.0014, 0.05) is None


def test_ev_signal_short_side_and_plan_tp():
    sig = make_ev_signal(0.1, 0.6, 100.0, 1.0, 2.0, 3.0, 0.0, 0.05)
    assert sig.direction == -1
    plan = plan_trade(sig, 10_000, 100.0, 1.0, RiskCfg(min_reward_risk=1.5))
    assert not isinstance(plan, Rejection)
    assert plan.take_profit == pytest.approx(97.0)


def test_round_trip_cost():
    assert round_trip_cost(0.0005, 2.0) == pytest.approx(0.0014)
```

(Check that `RiskCfg`, `plan_trade`, `Rejection` are already imported in `tests/test_risk.py`; add
imports only if missing.)

Commit: `Add expected-value signal rule`

---

## Phase 6 — Pluggable signal function in the backtester

File: `src/trader_bot/backtest/engine.py`

1. Imports: add `from collections.abc import Callable` and add `Signal` to the
   `trader_bot.risk.engine` import.
2. After `BAR = ...` add:
   ```python
   SignalFn = Callable[[pd.Series, float, float], Signal | None]  # (row, open price, atr)
   ```
3. Add a keyword parameter to `simulate`, after `diagnostics`:
   ```python
       signal_fn: SignalFn | None = None,
   ```
4. Inside the loop, replace the block from `row = preds.loc[t]` up to and including the
   `sig = None if halted else make_signal(...)` statement with:

```python
            row = preds.loc[t]
            if signal_fn is None:
                class_idx = [row["p_short"], row["p_flat"], row["p_long"]].index(
                    max(row["p_short"], row["p_flat"], row["p_long"])
                )
                class_name = ("short", "flat", "long")[class_idx]
                diag["top_class_counts"][class_name] += 1
                direction = (-1, 0, 1)[class_idx]
                if direction != 0:
                    diag["directional_top_class"] += 1
                    if float(row[f"p_{class_name}"]) >= confidence_threshold:
                        diag["confidence_pass"] += 1
                        if float(row["pred_ret"]) * direction > 0:
                            diag["return_agreement"] += 1
            if halted:
                diag["risk_halted"] += 1
            if halted:
                sig = None
            elif signal_fn is None:
                sig = make_signal(
                    row["p_short"], row["p_flat"], row["p_long"], row["pred_ret"],
                    confidence_threshold,
                )
            else:
                sig = signal_fn(row, float(bar["open"]), float(atr.loc[t]))
```

Everything after (`if sig is not None:` …) stays unchanged.

Add test to `tests/test_backtest_exec.py`:

```python
def test_simulate_accepts_custom_signal_fn():
    from trader_bot.risk.engine import Signal

    bars = _bars([[100, 101, 99.5, 100], [100, 106, 99.5, 105], [105, 105, 104, 105]])
    t = bars.index[0]
    preds = pd.DataFrame({"p_long_win": [0.9], "p_short_win": [0.1]}, index=[t])
    atr = pd.Series([1.0], index=[t])
    funding = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    _, trades = simulate(bars, preds, atr, funding, RISK, BT, 0.0, horizon=6,
                         initial_equity=10_000,
                         signal_fn=lambda row, price, a: Signal(1, 0.9, 0.05))
    assert len(trades) == 1 and trades.iloc[0]["pnl"] == pytest.approx(250.0)
```

Commit: `Allow custom signal functions in the backtester`

---

## Phase 7 — Barrier model, dataset, CV and walk-forward predictions

Create `src/trader_bot/models/barrier.py`:

```python
"""Two calibrated binary classifiers: P(long TP first) and P(short TP first)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from trader_bot.config.settings import StrategyConfig
from trader_bot.features.builder import model_columns
from trader_bot.labeling.barriers import make_barrier_labels
from trader_bot.models.cv import purged_walk_forward
from trader_bot.models.ensemble import DEFAULT_PARAMS
from trader_bot.models.train import CORE_FEATURES
from trader_bot.risk.engine import round_trip_cost

BARRIER_PRED_COLS = ["p_long_win", "p_short_win", "p_long_raw", "p_short_raw"]
MIN_CALIBRATION_ROWS = 200


@dataclass
class BarrierDataset:
    X: pd.DataFrame
    y_long: pd.Series
    y_short: pd.Series
    r_long: pd.Series
    r_short: pd.Series
    has_label: pd.Series
    columns: list[str]

    @property
    def labeled(self) -> np.ndarray:
        return np.flatnonzero(self.has_label.to_numpy())


def build_barrier_dataset(feats: pd.DataFrame, cfg: StrategyConfig) -> BarrierDataset:
    valid = feats.dropna(subset=[*CORE_FEATURES, "ctx_atr", "ctx_high", "ctx_low"])
    lab = make_barrier_labels(valid, cfg.labeling.horizon_candles, cfg.risk.atr_stop_mult,
                              cfg.barrier.tp_atr_mult)
    has = lab["r_long"].notna() & lab["r_short"].notna()
    return BarrierDataset(valid, lab["y_long"], lab["y_short"], lab["r_long"], lab["r_short"],
                          has, model_columns(valid))


def _binary_models(seed: int) -> list:
    return [
        LGBMClassifier(**DEFAULT_PARAMS["lgbm"], random_state=seed, verbose=-1),
        CatBoostClassifier(**DEFAULT_PARAMS["catboost"], loss_function="Logloss",
                           random_seed=seed, verbose=0),
    ]


class BarrierModel:
    """Per side: LightGBM + CatBoost averaged, then isotonic calibration on a purged tail."""

    def __init__(self, columns: list[str], purge: int, calib_frac: float = 0.2, seed: int = 42):
        self.columns = columns
        self.purge = purge
        self.calib_frac = calib_frac
        self.seed = seed
        self.models: dict[str, list] = {}
        self.calibrators: dict[str, IsotonicRegression | None] = {}

    def _raw(self, side: str, Xc: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(Xc)[:, 1] for m in self.models[side]], axis=0)

    def fit(self, X: pd.DataFrame, y_long: pd.Series, y_short: pd.Series) -> BarrierModel:
        """X rows must be time-ordered and fully labeled."""
        Xc = X[self.columns]
        n = len(Xc)
        cal_start = int(n * (1 - self.calib_frac))
        fit_end = cal_start - self.purge
        use_calibration = n - cal_start >= MIN_CALIBRATION_ROWS and fit_end > 0
        if not use_calibration:
            fit_end = n
        for side, y in (("long", y_long), ("short", y_short)):
            y_int = y.astype(int)
            if y_int.iloc[:fit_end].nunique() < 2:
                raise ValueError(f"{side} labels contain a single class")
            self.models[side] = _binary_models(self.seed)
            for m in self.models[side]:
                m.fit(Xc.iloc[:fit_end], y_int.iloc[:fit_end])
            if use_calibration:
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(self._raw(side, Xc.iloc[cal_start:]), y_int.iloc[cal_start:])
                self.calibrators[side] = iso
            else:
                self.calibrators[side] = None
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X[self.columns]
        out = pd.DataFrame(index=X.index)
        for side in ("long", "short"):
            raw = self._raw(side, Xc)
            iso = self.calibrators[side]
            out[f"p_{side}_raw"] = raw
            out[f"p_{side}_win"] = iso.predict(raw) if iso is not None else raw
        return out[BARRIER_PRED_COLS]


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    table = df.groupby("bin").agg(mean_pred=("p", "mean"), hit_rate=("y", "mean"),
                                  n=("y", "size"))
    return table.reset_index(drop=True).to_dict(orient="records")


def barrier_signal_metrics(P: pd.DataFrame, X: pd.DataFrame, y_long: pd.Series,
                           y_short: pd.Series, r_long: pd.Series, r_short: pd.Series,
                           cfg: StrategyConfig) -> dict:
    """Score probabilities and the EV trading rule on out-of-sample rows (all aligned)."""
    stop_mult, tp_mult = cfg.risk.atr_stop_mult, cfg.barrier.tp_atr_mult
    rr = tp_mult / stop_mult
    price = X["ctx_close"].to_numpy(dtype=float)
    atr = X["ctx_atr"].to_numpy(dtype=float)
    cost_r = round_trip_cost(cfg.backtest.taker_fee, cfg.backtest.slippage_bps) * price / (
        stop_mult * atr
    )
    pl, ps = P["p_long_win"].to_numpy(), P["p_short_win"].to_numpy()
    yl, ys = y_long.to_numpy(dtype=int), y_short.to_numpy(dtype=int)
    rl, rs = r_long.to_numpy(dtype=float), r_short.to_numpy(dtype=float)
    ev_long = pl * rr - (1 - pl) - cost_r
    ev_short = ps * rr - (1 - ps) - cost_r
    direction = np.where(ev_long >= ev_short, 1, -1)
    signal = np.maximum(ev_long, ev_short) >= cfg.barrier.min_ev_r
    realized = np.where(direction == 1, rl, rs) - cost_r
    hits = np.where(direction == 1, yl, ys)
    n_signals = int(signal.sum())
    return {
        "n_oos": len(P),
        "reward_risk": rr,
        "rr_meets_risk_floor": rr >= cfg.risk.min_reward_risk,
        "base_rate_long_win": float(yl.mean()),
        "base_rate_short_win": float(ys.mean()),
        "auc_long": _auc(yl, pl),
        "auc_short": _auc(ys, ps),
        "brier_long": float(brier_score_loss(yl, pl)),
        "brier_short": float(brier_score_loss(ys, ps)),
        "calibration_long": _calibration_table(yl, pl),
        "calibration_short": _calibration_table(ys, ps),
        "n_signals": n_signals,
        "n_long_signals": int((signal & (direction == 1)).sum()),
        "n_short_signals": int((signal & (direction == -1)).sum()),
        "signal_coverage": n_signals / len(P) if len(P) else 0.0,
        "hit_rate_on_signals": float(hits[signal].mean()) if n_signals else None,
        "mean_net_r_on_signals": float(realized[signal].mean()) if n_signals else None,
        "always_long_mean_net_r": float(np.mean(rl - cost_r)),
        "always_short_mean_net_r": float(np.mean(rs - cost_r)),
    }


def cv_evaluate_barrier(ds: BarrierDataset, cfg: StrategyConfig) -> dict:
    """Purged walk-forward CV of the barrier model and the EV signal rule."""
    lab = ds.labeled
    X = ds.X.iloc[lab]
    yl, ys = ds.y_long.iloc[lab], ds.y_short.iloc[lab]
    rl, rs = ds.r_long.iloc[lab], ds.r_short.iloc[lab]
    m, h = cfg.model, cfg.labeling.horizon_candles
    preds, folds = [], []
    for fold, (tr, te) in enumerate(
        purged_walk_forward(len(X), m.cv_splits, h, m.embargo_candles, m.min_train_candles),
        start=1,
    ):
        model = BarrierModel(ds.columns, purge=h + m.embargo_candles,
                             calib_frac=cfg.barrier.calib_frac).fit(
            X.iloc[tr], yl.iloc[tr], ys.iloc[tr]
        )
        p = model.predict(X.iloc[te])
        preds.append(p)
        fold_metrics = barrier_signal_metrics(p, X.iloc[te], yl.iloc[te], ys.iloc[te],
                                              rl.iloc[te], rs.iloc[te], cfg)
        folds.append({
            "fold": fold, "n_train": len(tr), "n_test": len(te),
            **{k: fold_metrics[k] for k in (
                "auc_long", "auc_short", "n_signals", "hit_rate_on_signals",
                "mean_net_r_on_signals")},
        })
    P = pd.concat(preds)
    idx = P.index
    metrics = barrier_signal_metrics(P, X.loc[idx], yl.loc[idx], ys.loc[idx], rl.loc[idx],
                                     rs.loc[idx], cfg)
    metrics["cv_folds"] = folds
    metrics["folds_with_positive_mean_r"] = sum(
        1 for f in folds
        if f["mean_net_r_on_signals"] is not None and f["mean_net_r_on_signals"] > 0
    )
    return metrics


def walk_forward_barrier_predictions(ds: BarrierDataset, cfg: StrategyConfig,
                                     start: pd.Timestamp) -> pd.DataFrame:
    """Out-of-sample predictions; models retrain every N candles on purged past data only."""
    m, h = cfg.model, cfg.labeling.horizon_candles
    first = max(int(ds.X.index.searchsorted(start)), m.min_train_candles)
    has = ds.has_label.to_numpy()
    out = []
    for s in range(first, len(ds.X), m.retrain_every_candles):
        e = min(s + m.retrain_every_candles, len(ds.X))
        idx = np.arange(0, max(s - h - m.embargo_candles, 0))
        idx = idx[has[idx]]
        if len(idx) < m.min_train_candles // 2:
            continue
        model = BarrierModel(ds.columns, purge=h + m.embargo_candles,
                             calib_frac=cfg.barrier.calib_frac).fit(
            ds.X.iloc[idx], ds.y_long.iloc[idx], ds.y_short.iloc[idx]
        )
        out.append(model.predict(ds.X.iloc[s:e]))
    return pd.concat(out) if out else pd.DataFrame(columns=BARRIER_PRED_COLS)
```

Create `tests/test_barrier_model.py` (synthetic; keep it fast):

```python
import numpy as np
import pandas as pd

from trader_bot.config.settings import StrategyConfig
from trader_bot.models.barrier import (
    BARRIER_PRED_COLS, BarrierDataset, BarrierModel, cv_evaluate_barrier,
)


def _synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    x = rng.normal(size=n)
    X = pd.DataFrame({"f1": x, "f2": rng.normal(size=n),
                      "ctx_close": 100.0, "ctx_atr": 1.0}, index=idx)
    y_long = pd.Series((x + rng.normal(scale=0.5, size=n) > 0.5).astype(float), index=idx)
    y_short = pd.Series((-x + rng.normal(scale=0.5, size=n) > 0.5).astype(float), index=idx)
    r_long = y_long * 1.5 - (1 - y_long)
    r_short = y_short * 1.5 - (1 - y_short)
    has = pd.Series(True, index=idx)
    return BarrierDataset(X, y_long, y_short, r_long, r_short, has, ["f1", "f2"])


def test_barrier_model_outputs_probabilities():
    ds = _synthetic()
    model = BarrierModel(ds.columns, purge=12, calib_frac=0.3).fit(ds.X, ds.y_long, ds.y_short)
    pred = model.predict(ds.X)
    assert list(pred.columns) == BARRIER_PRED_COLS
    assert ((pred >= 0) & (pred <= 1)).all().all()
    assert model.calibrators["long"] is not None
    assert pred["p_long_win"].corr(ds.X["f1"]) > 0.3


def test_cv_evaluate_barrier_reports_trading_metrics():
    ds = _synthetic(n=1200)
    cfg = StrategyConfig()
    cfg.model = cfg.model.model_copy(update={"min_train_candles": 600, "cv_splits": 2})
    metrics = cv_evaluate_barrier(ds, cfg)
    for key in ("auc_long", "auc_short", "n_signals", "mean_net_r_on_signals",
                "folds_with_positive_mean_r", "calibration_long"):
        assert key in metrics
    assert metrics["auc_long"] > 0.6
```

Verify: `uv run pytest -q tests/test_barrier_model.py` (may take ~10–30 s), then full suite + ruff.

Commit: `Add calibrated barrier model with purged CV`

---

## Phase 8 — Research commands: barrier experiment and barrier backtest

### 8.1 Bootstrap confidence interval for Sharpe

File: `src/trader_bot/backtest/metrics.py` — add:

```python
def bootstrap_sharpe_ci(equity: pd.Series, n_boot: int = 1000, block: int = 10,
                        seed: int = 0) -> list[float] | None:
    """90% block-bootstrap interval of the annualised daily Sharpe ratio."""
    rets = equity.resample("1D").last().dropna().pct_change().dropna().to_numpy()
    n = len(rets)
    if n < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sharpes = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([rets[s:s + block] for s in starts])[:n]
        sd = sample.std(ddof=1)
        sharpes.append(sample.mean() / sd * np.sqrt(365) if sd > 0 else 0.0)
    low, high = np.percentile(sharpes, [5, 95])
    return [float(low), float(high)]
```

Test in `tests/test_backtest_exec.py`:

```python
def test_bootstrap_sharpe_ci_brackets_point_estimate():
    from trader_bot.backtest.metrics import bootstrap_sharpe_ci

    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=400, freq="1D", tz="UTC")
    equity = pd.Series(10_000 * np.cumprod(1 + rng.normal(0.001, 0.01, 400)), index=idx)
    low, high = bootstrap_sharpe_ci(equity)
    point = compute_metrics(equity, pd.DataFrame(columns=["pnl"]))["sharpe"]
    assert low < point < high
```

### 8.2 Shared segment simulator

Create `src/trader_bot/backtest/segments.py`:

```python
"""Simulate a multi-symbol portfolio over a date segment with any signal function."""
from __future__ import annotations

import pandas as pd

from trader_bot.backtest.engine import BAR, SignalFn, simulate
from trader_bot.backtest.metrics import bootstrap_sharpe_ci, compute_metrics, passes_gate
from trader_bot.config.settings import StrategyConfig


def simulate_segment(prepared: dict[str, dict], cfg: StrategyConfig, seg_start: pd.Timestamp,
                     seg_end: pd.Timestamp, signal_fn: SignalFn) -> dict:
    """prepared[symbol] = {"preds", "bars", "atr", "funding"}. Entries that cannot finish their
    horizon before seg_end are dropped so segments never share trades."""
    share = cfg.backtest.initial_equity / len(cfg.symbols)
    tradable_end = seg_end - cfg.labeling.horizon_candles * BAR
    curves, trades_all, per_symbol = [], [], {}
    for symbol, item in prepared.items():
        preds = item["preds"]
        preds = preds.loc[(preds.index >= seg_start) & (preds.index <= tradable_end)]
        if preds.empty:
            continue
        bars = item["bars"].loc[(item["bars"].index >= seg_start) & (item["bars"].index < seg_end)]
        diagnostics: dict = {}
        equity, trades = simulate(
            bars, preds, item["atr"].reindex(preds.index), item["funding"], cfg.risk,
            cfg.backtest, 0.0, cfg.labeling.horizon_candles, share, diagnostics,
            signal_fn=signal_fn,
        )
        curves.append(equity.rename(symbol))
        if not trades.empty:
            trades_all.append(trades.assign(symbol=symbol))
        per_symbol[symbol] = {**compute_metrics(equity, trades),
                              "signal_diagnostics": diagnostics}
    if not curves:
        return {"portfolio": {"n_trades": 0, "gate_passed": False}, "per_symbol": {}}
    portfolio = pd.concat(curves, axis=1).ffill().bfill().sum(axis=1)
    trades_df = (pd.concat(trades_all, ignore_index=True) if trades_all
                 else pd.DataFrame(columns=["pnl"]))
    metrics = compute_metrics(portfolio, trades_df)
    metrics["gate_passed"] = passes_gate(metrics)
    metrics["sharpe_ci_90"] = bootstrap_sharpe_ci(portfolio)
    return {"portfolio": metrics, "per_symbol": per_symbol}


def segments_for(start: pd.Timestamp, end: pd.Timestamp,
                 holdout_start: pd.Timestamp | None) -> dict[str, tuple]:
    if holdout_start is None:
        return {"all": (start, end)}
    return {"validation": (start, holdout_start), "holdout": (holdout_start, end)}
```

### 8.3 Barrier backtest runner

Create `src/trader_bot/backtest/run_barrier.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy.engine import Engine

from trader_bot.backtest.engine import BAR
from trader_bot.backtest.segments import segments_for, simulate_segment
from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.models.barrier import build_barrier_dataset, walk_forward_barrier_predictions
from trader_bot.risk.engine import make_ev_signal, round_trip_cost

log = structlog.get_logger()


def ev_signal_fn(cfg: StrategyConfig):
    cost = round_trip_cost(cfg.backtest.taker_fee, cfg.backtest.slippage_bps)

    def fn(row: pd.Series, price: float, atr: float):
        return make_ev_signal(float(row["p_long_win"]), float(row["p_short_win"]), price, atr,
                              cfg.risk.atr_stop_mult, cfg.barrier.tp_atr_mult, cost,
                              cfg.barrier.min_ev_r)
    return fn


def run_barrier_backtest(engine: Engine, cfg: StrategyConfig, out_dir: Path,
                         holdout_start: pd.Timestamp | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(cfg.backtest.start, tz="UTC")
    prepared = {}
    for symbol in cfg.symbols:
        feats, raw = features_for_symbol(engine, symbol, cfg)
        ds = build_barrier_dataset(feats, cfg)
        preds = walk_forward_barrier_predictions(ds, cfg, start)
        if preds.empty:
            log.warning("barrier_backtest.no_predictions", symbol=symbol)
            continue
        preds.to_csv(out_dir / f"{symbol}_barrier_predictions.csv")
        prepared[symbol] = {"preds": preds, "bars": raw["k4"], "atr": ds.X["ctx_atr"],
                            "funding": raw["funding"]["rate"]}
    if not prepared:
        raise RuntimeError("no barrier predictions; is there enough history in the DB?")
    end = max(item["bars"].index.max() for item in prepared.values()) + BAR
    signal_fn = ev_signal_fn(cfg)
    report = {
        "settings": {
            "signal_mode": "barrier",
            "horizon_candles": cfg.labeling.horizon_candles,
            "atr_stop_mult": cfg.risk.atr_stop_mult,
            "tp_atr_mult": cfg.barrier.tp_atr_mult,
            "min_ev_r": cfg.barrier.min_ev_r,
            "min_reward_risk": cfg.risk.min_reward_risk,
            "taker_fee": cfg.backtest.taker_fee,
            "slippage_bps": cfg.backtest.slippage_bps,
            "holdout_start": str(holdout_start) if holdout_start is not None else None,
        },
        "segments": {
            name: simulate_segment(prepared, cfg, a, b, signal_fn)
            for name, (a, b) in segments_for(start, end, holdout_start).items()
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
```

### 8.4 Barrier CV experiment (selection uses pre-holdout data only)

Create `src/trader_bot/models/barrier_experiment.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy.engine import Engine

from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.models.barrier import build_barrier_dataset, cv_evaluate_barrier

log = structlog.get_logger()
SUMMARY_KEYS = ("reward_risk", "rr_meets_risk_floor", "auc_long", "auc_short", "n_signals",
                "signal_coverage", "hit_rate_on_signals", "mean_net_r_on_signals",
                "folds_with_positive_mean_r", "always_long_mean_net_r")


def run_barrier_experiment(engine: Engine, cfg: StrategyConfig, out_dir: Path,
                           holdout_start: pd.Timestamp, tp_atr_mults: list[float],
                           horizons: list[int]) -> dict:
    """CV grid over take-profit multiple and horizon. Rows at/after holdout_start are removed
    before labels are built, so no holdout information reaches selection."""
    report: dict = {"holdout_start": str(holdout_start), "symbols": {}}
    for symbol in cfg.symbols:
        feats, _ = features_for_symbol(engine, symbol, cfg)
        feats = feats.loc[feats.index < holdout_start]
        results = []
        for horizon in horizons:
            for tp in tp_atr_mults:
                variant = cfg.model_copy(update={
                    "labeling": cfg.labeling.model_copy(update={"horizon_candles": horizon}),
                    "barrier": cfg.barrier.model_copy(update={"tp_atr_mult": tp}),
                })
                metrics = cv_evaluate_barrier(build_barrier_dataset(feats, variant), variant)
                results.append({"horizon_candles": horizon, "tp_atr_mult": tp,
                                "metrics": metrics})
                log.info("barrier_experiment.variant", symbol=symbol, horizon=horizon, tp=tp,
                         **{k: metrics[k] for k in SUMMARY_KEYS})
        report["symbols"][symbol] = results
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
```

### 8.5 CLI commands

File: `src/trader_bot/cli.py`. Add two commands **after** `signal_study` (copy the existing
date-parsing style from `signal_study`):

```python
def _parse_utc(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{name} must be a date or timestamp") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@app.command("barrier-experiment")
def barrier_experiment(
    holdout_start: str = typer.Option("2025-10-01", help="Rows from this date are excluded"),
    tp_atr_mults: str = typer.Option("2.0,3.0,4.0", help="Comma-separated take-profit ATR mults"),
    horizons: str = typer.Option("3,6,12", help="Comma-separated horizons in candles"),
) -> None:
    """Purged-CV grid for barrier labels + EV rule, using only pre-holdout data."""
    import pandas as pd

    from trader_bot.models.barrier_experiment import SUMMARY_KEYS, run_barrier_experiment

    _, cfg, engine = _ctx()
    try:
        tps = [float(v) for v in tp_atr_mults.split(",")]
        hs = [int(v) for v in horizons.split(",")]
    except ValueError as exc:
        raise typer.BadParameter("use comma-separated numbers") from exc
    if any(v <= 0 for v in tps) or any(v <= 0 for v in hs):
        raise typer.BadParameter("values must be positive")
    holdout = pd.Timestamp(_parse_utc(holdout_start, "holdout_start"))
    out = Path("artifacts/experiments") / datetime.now(UTC).strftime("barrier-%Y%m%dT%H%M%SZ")
    report = run_barrier_experiment(engine, cfg, out, holdout, tps, hs)
    summary = {
        symbol: [{"horizon_candles": r["horizon_candles"], "tp_atr_mult": r["tp_atr_mult"],
                  **{k: r["metrics"][k] for k in SUMMARY_KEYS}} for r in results]
        for symbol, results in report["symbols"].items()
    }
    typer.echo(json.dumps({"report": str(out / "report.json"), "summary": summary},
                          indent=2, default=str))


@app.command("barrier-backtest")
def barrier_backtest(
    holdout_start: str | None = typer.Option(None, help="Split report into validation/holdout"),
    out: Path | None = typer.Option(None, help="Output directory"),
) -> None:
    """Walk-forward backtest with barrier labels and the expected-value signal rule."""
    import pandas as pd

    from trader_bot.backtest.run_barrier import run_barrier_backtest

    _, cfg, engine = _ctx()
    if cfg.barrier.tp_atr_mult / cfg.risk.atr_stop_mult < cfg.risk.min_reward_risk:
        raise typer.BadParameter("barrier.tp_atr_mult / atr_stop_mult is below min_reward_risk")
    holdout = pd.Timestamp(_parse_utc(holdout_start, "holdout_start")) if holdout_start else None
    out = out or Path("artifacts/backtest_barrier")
    report = run_barrier_backtest(engine, cfg, out, holdout)
    typer.echo(json.dumps({"report": str(out / "report.json"),
                           "segments": {k: v["portfolio"] for k, v in report["segments"].items()}},
                          indent=2, default=str))
```

Note: `barrier-backtest` uses `cfg.barrier.*` from `config/strategy.yaml` even while
`signal_mode: classic`; it never affects live trading.

Verify:
```bash
uv run pytest -q && uv run ruff check src tests
uv run trader-bot barrier-experiment --help
uv run trader-bot barrier-backtest --help
```

Commit: `Add barrier experiment and backtest commands`

---

## Phase 9 — Rule-based baselines (same simulator, same costs)

Create `src/trader_bot/backtest/baselines.py`:

```python
"""Simple trailing-only strategies evaluated with the same simulator, sizing and costs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from trader_bot.backtest.engine import BAR
from trader_bot.backtest.segments import segments_for, simulate_segment
from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.risk.engine import Signal

BASELINES = ("always_long", "momentum_42", "breakout_20", "funding_contrarian")


def baseline_direction(feats: pd.DataFrame, name: str) -> pd.Series:
    """+1 / -1 / 0 per decision time, using only data available at that time."""
    close = feats["ctx_close"]
    if name == "always_long":
        return pd.Series(1, index=feats.index)
    if name == "momentum_42":
        return np.sign(feats["ret_42"]).fillna(0).astype(int)
    if name == "breakout_20":
        upper = feats["ctx_high"].rolling(20).max().shift(1)
        lower = feats["ctx_low"].rolling(20).min().shift(1)
        return pd.Series(np.where(close > upper, 1, np.where(close < lower, -1, 0)),
                         index=feats.index)
    if name == "funding_contrarian":
        if "funding_rate" not in feats:
            return pd.Series(0, index=feats.index)
        f = feats["funding_rate"]
        z = (f - f.rolling(180).mean()) / f.rolling(180).std()
        return pd.Series(np.where(z > 2, -1, np.where(z < -2, 1, 0)), index=feats.index)
    raise ValueError(f"unknown baseline {name}")


def baseline_signal_fn(cfg: StrategyConfig):
    """TP placed exactly at the configured minimum reward/risk."""
    tp_mult = cfg.risk.min_reward_risk * cfg.risk.atr_stop_mult

    def fn(row: pd.Series, price: float, atr: float):
        d = int(row["direction"])
        if d == 0 or price <= 0 or atr <= 0:
            return None
        return Signal(d, 1.0, d * tp_mult * atr / price * (1 + 1e-9))
    return fn


def run_baselines(engine: Engine, cfg: StrategyConfig, out_dir: Path,
                  holdout_start: pd.Timestamp | None = None) -> dict:
    start = pd.Timestamp(cfg.backtest.start, tz="UTC")
    data = {}
    for symbol in cfg.symbols:
        feats, raw = features_for_symbol(engine, symbol, cfg)
        data[symbol] = (feats, raw)
    end = max(raw["k4"].index.max() for _, raw in data.values()) + BAR
    report: dict = {"baselines": {}}
    for name in BASELINES:
        prepared = {}
        for symbol, (feats, raw) in data.items():
            direction = baseline_direction(feats, name)
            preds = pd.DataFrame({"direction": direction}).loc[lambda d: d.index >= start]
            prepared[symbol] = {"preds": preds, "bars": raw["k4"], "atr": feats["ctx_atr"],
                                "funding": raw["funding"]["rate"]}
        report["baselines"][name] = {
            seg: simulate_segment(prepared, cfg, a, b, baseline_signal_fn(cfg))
            for seg, (a, b) in segments_for(start, end, holdout_start).items()
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
```

CLI in `src/trader_bot/cli.py`:

```python
@app.command()
def baselines(
    holdout_start: str | None = typer.Option(None, help="Split report into validation/holdout"),
    out: Path = typer.Option(Path("artifacts/baselines"), help="Output directory"),
) -> None:
    """Rule-based baselines with the same simulator, sizing and costs as the ML backtest."""
    import pandas as pd

    from trader_bot.backtest.baselines import run_baselines

    _, cfg, engine = _ctx()
    holdout = pd.Timestamp(_parse_utc(holdout_start, "holdout_start")) if holdout_start else None
    report = run_baselines(engine, cfg, out, holdout)
    summary = {name: {seg: r["portfolio"] for seg, r in segs.items()}
               for name, segs in report["baselines"].items()}
    typer.echo(json.dumps(summary, indent=2, default=str))
```

Tests — create `tests/test_baselines.py`:

```python
import numpy as np
import pandas as pd

from trader_bot.backtest.baselines import baseline_direction


def _feats(n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = pd.Series(np.linspace(100, 130, n), index=idx)
    return pd.DataFrame({"ctx_close": close, "ctx_high": close + 0.5, "ctx_low": close - 0.5,
                         "ret_42": np.log(close).diff(42), "funding_rate": 0.0001}, index=idx)


def test_breakout_uses_only_previous_highs():
    d = baseline_direction(_feats(), "breakout_20")
    assert d.iloc[:20].eq(0).all()
    assert d.iloc[-1] == 1   # steadily rising close exceeds prior 20-row high


def test_baselines_unchanged_when_future_removed():
    feats = _feats()
    for name in ("momentum_42", "breakout_20", "funding_contrarian"):
        full = baseline_direction(feats, name)
        cut = baseline_direction(feats.iloc[:40], name)
        pd.testing.assert_series_equal(full.iloc[:40], cut, check_names=False,
                                       check_dtype=False)
```

Commit: `Add rule-based baselines`

---

## Phase 10 — Documentation

Update (short, factual, no claims of profitability):

- `docs/data-and-ml.md`: add a section **"Barrier mode (research)"** describing labels
  (stop/TP/time in R units, stop checked first), the two calibrated classifiers, the EV rule
  `EV = p·RR − (1−p) − cost_R` (time exits counted as losses → conservative), the commands
  `barrier-experiment`, `barrier-backtest`, `baselines`, and that selection must use
  `barrier-experiment` (pre-holdout) before a single `barrier-backtest --holdout-start` run.
  Replace the line "Not implemented: probability calibration." with "Probability calibration is
  implemented only in barrier mode (isotonic, purged tail of each training window)."
- `docs/architecture.md`: add `labeling/barriers.py`, `models/barrier.py`,
  `models/barrier_experiment.py`, `backtest/segments.py`, `backtest/run_barrier.py`,
  `backtest/baselines.py` to the module map.
- `README.md`: add the three commands to Quick start with a `# research` comment.
- `GAPS.md`: under "Gaps to Close" item 1 (baselines), note that `trader-bot baselines` now exists.

Commit: `Document barrier mode and baselines`

---

## Phase 11 — Run the research protocol (human checkpoint)

Requires the database to be populated (`init-db`, `backfill`). Run exactly in this order and
**save every report path**:

```bash
uv run trader-bot baselines --holdout-start 2025-10-01
uv run trader-bot barrier-experiment --holdout-start 2025-10-01
```

Selection rule (apply by reading `artifacts/experiments/barrier-*/report.json`; do not
re-run with different grids to "find" a winner):

- candidate must have `rr_meets_risk_floor: true` for **both** symbols,
- `folds_with_positive_mean_r >= 4` (of 5) for **both** symbols,
- `n_signals >= 50` per symbol,
- `mean_net_r_on_signals` greater than `always_long_mean_net_r` (beats naive exposure).
- Among passing candidates choose the highest average `mean_net_r_on_signals`.

If **no** candidate passes: stop, write the result into `ISSUES.md`, and do not proceed.

If one passes: set its `horizon_candles` and `barrier.tp_atr_mult` **only in a copy** of the
config (e.g. `STRATEGY_CONFIG=config/strategy_barrier.yaml`, with `signal_mode: barrier`), then
run **once**:

```bash
STRATEGY_CONFIG=config/strategy_barrier.yaml \
  uv run trader-bot barrier-backtest --holdout-start 2025-10-01
```

Record validation and holdout metrics, including `sharpe_ci_90`, trade counts, and baseline
comparison, in `ISSUES.md`. **STOP and hand the results to a human.** Do not start Phase 12
without explicit human approval.

---

## Phase 12 — Live wiring (only after human approval)

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

## Backlog (optional, each is a separate small task; same rules apply)

Implement only after Phase 11, one at a time, each evaluated with `barrier-experiment` on
pre-holdout data. Never re-use the holdout for selection.

| ID | Change | Where | Acceptance |
|---|---|---|---|
| D1 | **Label-uniqueness sample weights**: weight_i = mean over bars in [i+1, i+h] of 1/(number of labels covering that bar); pass `sample_weight` to both `fit` calls in `BarrierModel`. | `models/barrier.py` | Unit test: non-overlapping labels → all weights 1.0; fully overlapping → 1/h in the middle. |
| D2 | **Stationary features**: replace raw `oi_value` with `oi_z_180 = (oi - rolling_mean_180) / rolling_std_180` computed on the 4h grid after the as-of join; z-score `top_ls_*`, `global_ls` the same way. Keep old columns only if a config flag is on. | `features/builder.py` | Leakage test in `tests/test_features.py` still passes; new columns exist. |
| D3 | **Pooled BTC+ETH model**: concatenate both datasets sorted by time, add `sym_is_eth` (0/1) feature, use the time-sorted union for purged CV (purge by timestamp, not row). | new `models/pooled.py` | CV report per symbol computed from pooled OOS predictions. |
| D4 | **Regularisation**: LightGBM `num_leaves=15, min_child_samples=100, n_estimators=300`, CatBoost `depth=4`; compare via experiment. | `models/barrier.py` constants | Experiment report includes both settings. |
| E1 | **Cross-asset features**: for ETH add BTC `ret_1/6/42`, `vol_42`; add `eth_btc_ret_42`. Load the other symbol's 4h klines in `features/pipeline.py`. | `features/builder.py`, `features/pipeline.py` | Leakage test extended to cross-asset columns. |
| E2 | **Calendar features**: `bar_hour` (0,4,…,20) and `dow` from the decision timestamp. | `features/builder.py` | Trivial unit test. |
| E3 | **Funding z-score** and **OI–price divergence** (`oi_chg_24h - ret_6`). | `features/builder.py` | Columns present; leakage test passes. |
| H1 | **Backfill from 2019-09** for BTCUSDT/ETHUSDT 4h/1h/1d klines + funding to enlarge training data. | operations only | `report_gaps` clean. |
| G1 | **Nested CV for `tune`**: tune only on the training part of each outer fold. | `models/train.py` | Reported CV metrics come only from outer test folds. |

---

## Definition of done (Phases 1–10)

- [ ] `uv run pytest -q` passes; test count increased by ≥ 20.
- [ ] `uv run ruff check src tests` clean.
- [ ] `config/strategy.yaml` still has `signal_mode: classic`; classic `backtest` output unchanged
      except TP-exit slippage.
- [ ] `barrier-experiment`, `barrier-backtest`, `baselines` and `backtest --registry-params`
      appear in `uv run trader-bot --help`.
- [ ] Docs updated as in Phase 10.
- [ ] No changes to `risk/limits.py`, `risk/kill_switch.py`, `execution/`.
