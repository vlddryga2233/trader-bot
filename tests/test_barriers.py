import numpy as np
import pandas as pd
import pytest

from trader_bot.labeling.barriers import limit_entry_outcomes, make_barrier_labels


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
    assert lab["r_long"].iloc[0] == pytest.approx(1.5)
    assert lab["y_long"].iloc[0] == 1.0
    assert lab["r_short"].iloc[0] == pytest.approx(-1.0)
    assert lab["y_short"].iloc[0] == 0.0
    assert lab["r_long"].iloc[3] == pytest.approx(0.0)
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
