import numpy as np
import pandas as pd

from trader_bot.config.settings import load_strategy
from trader_bot.features.daily import daily_features
from trader_bot.models.train import CORE_FEATURES


def _kd(n=300, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC", name="open_time")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    frame = pd.DataFrame({"close": close}, index=idx)
    frame["open"] = frame["close"].shift(1).fillna(frame["close"])
    frame["high"] = frame[["open", "close"]].max(axis=1) * 1.01
    frame["low"] = frame[["open", "close"]].min(axis=1) * 0.99
    frame["volume"] = rng.uniform(1, 10, n)
    frame["taker_buy_base"] = frame["volume"] * rng.uniform(0.4, 0.6, n)
    return frame


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


def test_strategy_file_cannot_enable_daily_interval(tmp_path):
    cfg = tmp_path / "strategy.yaml"
    cfg.write_text("decision_interval: 1d\n")
    try:
        load_strategy(cfg)
    except ValueError as exc:
        assert "research-only" in str(exc)
    else:
        raise AssertionError("daily interval must remain research-only in strategy files")
