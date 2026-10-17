import numpy as np
import pandas as pd

from trader_bot.backtest.baselines import baseline_direction


def _feats(n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = pd.Series(np.linspace(100, 130, n), index=idx)
    return pd.DataFrame({"ctx_close": close, "ctx_high": close + 0.5, "ctx_low": close - 0.5,
                         "ret_42": np.log(close).diff(42), "funding_rate": 0.0001}, index=idx)


def test_breakout_uses_only_previous_highs():
    direction = baseline_direction(_feats(), "breakout_20")
    assert direction.iloc[:20].eq(0).all()
    assert direction.iloc[-1] == 1


def test_baselines_unchanged_when_future_removed():
    feats = _feats()
    for name in ("momentum_42", "breakout_20", "funding_contrarian"):
        full = baseline_direction(feats, name)
        cut = baseline_direction(feats.iloc[:40], name)
        pd.testing.assert_series_equal(full.iloc[:40], cut, check_names=False,
                                       check_dtype=False)
