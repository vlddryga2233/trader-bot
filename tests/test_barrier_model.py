import numpy as np
import pandas as pd

from trader_bot.config.settings import StrategyConfig
from trader_bot.models.barrier import (
    BARRIER_PRED_COLS,
    BarrierDataset,
    BarrierModel,
    cv_evaluate_barrier,
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
