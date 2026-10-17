import numpy as np
import pandas as pd

from trader_bot.config.settings import StrategyConfig
from trader_bot.models.barrier import (
    BARRIER_PRED_COLS,
    BarrierDataset,
    BarrierModel,
    _t_stat,
    barrier_signal_metrics,
    cv_evaluate_barrier,
    non_overlapping,
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


def test_small_model_option_fits():
    ds = _synthetic()
    model = BarrierModel(ds.columns, purge=12, calib_frac=0.3, size="small").fit(
        ds.X, ds.y_long, ds.y_short)
    assert model.models["long"][0].get_params()["num_leaves"] == 15


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
