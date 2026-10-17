import numpy as np
import pandas as pd
import pytest

from trader_bot.config.settings import FeatureGroups
from trader_bot.features.builder import build_features, model_columns

N = 400


def _klines(freq: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC", name="open_time")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    df = pd.DataFrame({"close": close}, index=idx)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * 1.002
    df["low"] = df[["open", "close"]].min(axis=1) * 0.998
    df["volume"] = rng.uniform(1, 10, n)
    df["taker_buy_base"] = df["volume"] * rng.uniform(0.4, 0.6, n)
    return df


def _inputs():
    k4 = _klines("4h", N, 1)
    k1 = _klines("1h", N * 4, 2)
    kd = _klines("1D", N // 6 + 5, 3)
    fidx = pd.date_range("2024-01-01", periods=N // 2, freq="8h", tz="UTC", name="ts")
    funding = pd.DataFrame({"rate": np.linspace(-1e-4, 1e-4, len(fidx))}, index=fidx)
    midx = pd.date_range("2024-01-01", periods=N * 48, freq="5min", tz="UTC", name="ts")
    rng = np.random.default_rng(4)
    metrics = pd.DataFrame({
        "oi": rng.uniform(1, 2, len(midx)), "oi_value": rng.uniform(1e9, 2e9, len(midx)),
        "top_ls_accounts": 1.0, "top_ls_positions": 1.0, "global_ls": 1.0, "taker_ls_vol": 1.0,
    }, index=midx)
    empty = pd.DataFrame()
    return k4, k1, kd, funding, metrics, empty, empty


def _build(k4, k1, kd, funding, metrics, liq, onchain):
    return build_features(k4, k1, kd, funding, metrics, liq, onchain,
                          FeatureGroups(price=True, derivatives=True))


def test_index_is_candle_close_time():
    feats = _build(*_inputs())
    k4 = _inputs()[0]
    assert feats.index[0] == k4.index[0] + pd.Timedelta(hours=4)


def test_ctx_high_low_are_not_model_inputs():
    feats = _build(*_inputs())
    assert "ctx_high" in feats.columns and "ctx_low" in feats.columns
    assert "ctx_high" not in model_columns(feats)
    assert "ctx_low" not in model_columns(feats)


def test_no_future_leakage_when_future_is_truncated():
    """Features at time t must be identical if all data after t is removed."""
    full = _build(*_inputs())
    t = full.index[250]
    k4, k1, kd, funding, metrics, liq, oc = _inputs()
    cut = _build(
        k4[k4.index + pd.Timedelta(hours=4) <= t],
        k1[k1.index + pd.Timedelta(hours=1) <= t],
        kd[kd.index + pd.Timedelta(days=1) <= t],
        funding[funding.index <= t],
        metrics[metrics.index <= t],
        liq, oc,
    )
    cols = model_columns(full)
    pd.testing.assert_series_equal(full.loc[t, cols], cut.loc[t, cols], check_names=False)


def test_daily_candle_not_visible_before_it_closes():
    feats = _build(*_inputs())
    kd = _inputs()[2]
    t = feats.index[30]  # decision time
    visible = kd[kd.index + pd.Timedelta(days=1) <= t]
    expected = np.log(visible["close"]).diff().iloc[-1]
    assert feats.loc[t, "d1_ret_1"] == pytest.approx(expected)


def test_onchain_publication_lag_applied():
    from trader_bot.features.builder import onchain_features

    base = pd.DataFrame({"x": 1.0}, index=pd.date_range("2024-01-02", periods=12, freq="4h", tz="UTC"))
    oc = pd.DataFrame({"metric": ["nupl"], "value": [0.5]},
                      index=pd.DatetimeIndex(["2024-01-02 00:00"], tz="UTC"))
    out = onchain_features(base, oc, lag_hours=24)
    before = out.loc[out.index < pd.Timestamp("2024-01-03", tz="UTC"), "oc_nupl"]
    assert before.isna().all()  # value stamped Jan 2 is only usable from Jan 3 00:00
    assert (out.loc[out.index >= pd.Timestamp("2024-01-03", tz="UTC"), "oc_nupl"] == 0.5).all()
    base2 = pd.DataFrame({"x": 1.0}, index=pd.date_range("2024-01-03", periods=3, freq="4h", tz="UTC"))
    assert (onchain_features(base2, oc, 24)["oc_nupl"] == 0.5).all()


def test_metrics_changes_use_elapsed_time_and_reject_zero_oi():
    from trader_bot.features.builder import _time_lagged_change

    idx = pd.date_range("2024-01-01", periods=25, freq="5min", tz="UTC")
    oi = pd.Series(100.0, index=idx)
    oi.iloc[0] = 0.0
    oi.iloc[12] = 200.0
    change = _time_lagged_change(oi.where(oi > 0), 12)
    assert np.isnan(change.iloc[12])  # zero baseline is invalid
    assert change.iloc[24] == pytest.approx(-0.5)

    # The baseline is 20 minutes before the requested time, outside the 10-minute bound.
    irregular_idx = pd.DatetimeIndex([idx[0], idx[16]])
    irregular = pd.Series([100.0, 200.0], index=irregular_idx)
    irregular_change = _time_lagged_change(irregular, 12)
    assert np.isnan(irregular_change.iloc[-1])


def test_metrics_asof_does_not_carry_stale_observations():
    from trader_bot.features.builder import derivatives_features

    base_idx = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 04:00", "2024-01-01 08:00"], tz="UTC"
    )
    base = pd.DataFrame({"price": [1.0, 2.0, 3.0]}, index=base_idx)
    metrics_idx = pd.date_range("2024-01-01", periods=61, freq="5min", tz="UTC")
    metrics_idx = metrics_idx.append(pd.DatetimeIndex(["2024-01-01 04:05"], tz="UTC"))
    metrics = pd.DataFrame(
        {
            "oi_value": 100.0,
            "top_ls_accounts": 1.0,
            "top_ls_positions": 1.0,
            "global_ls": 1.0,
            "taker_ls_vol": 1.0,
        },
        index=metrics_idx,
    ).sort_index()
    out = derivatives_features(base, pd.DataFrame(), metrics)
    assert out.loc[base_idx[1], "oi_value"] == 100.0
    assert np.isnan(out.loc[base_idx[2], "oi_value"])
