"""Leakage-safe feature construction.

Every row is indexed by the decision time t (close of a 4h candle). A feature may only use
data whose availability time is <= t. Source features are computed on their native grid with
trailing-only operations, stamped with the time they become available, then joined with
merge_asof(direction="backward").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader_bot.config.settings import FeatureGroups

PRIMARY = pd.Timedelta(hours=4)
METRICS_MAX_AGE = pd.Timedelta(minutes=15)
METRICS_LAG_TOLERANCE = pd.Timedelta(minutes=10)
METRICS_SAMPLE_INTERVAL = pd.Timedelta(minutes=5)
CTX_PREFIX = "ctx_"


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def _stamp_closed(df: pd.DataFrame, length: pd.Timedelta) -> pd.DataFrame:
    """Re-index candles by close time (= availability time)."""
    out = df.copy()
    out.index = out.index + length
    out.index.name = "avail"
    return out


def _asof(
    base: pd.DataFrame, feats: pd.DataFrame, tolerance: pd.Timedelta | None = None
) -> pd.DataFrame:
    """Attach the latest feature row available at each base timestamp."""
    if feats.empty:
        return base
    feats = feats[~feats.index.duplicated(keep="last")].sort_index()
    left = base.reset_index().rename(columns={base.index.name or "index": "t"})
    right = feats.reset_index().rename(columns={feats.index.name or "index": "avail"})
    left["t"] = left["t"].astype("datetime64[ns, UTC]")
    right["avail"] = right["avail"].astype("datetime64[ns, UTC]")
    merged = pd.merge_asof(left.sort_values("t"), right.sort_values("avail"),
                           left_on="t", right_on="avail", direction="backward",
                           tolerance=tolerance)
    merged = merged.drop(columns=["avail"]).set_index("t")
    merged.index.name = base.index.name
    return merged


def _time_lagged_change(
    values: pd.Series,
    periods: int,
    *,
    percent: bool = True,
) -> pd.Series:
    """Compare with the latest observation near the requested elapsed-time lag.

    The source is expected on a 5-minute grid. Rows are not treated as fixed-duration
    intervals: if the baseline observation is more than 10 minutes before its target,
    the change is unavailable.
    """
    values = values.sort_index()
    idx = values.index
    target = idx - periods * METRICS_SAMPLE_INTERVAL
    positions = idx.searchsorted(target, side="right") - 1
    valid = positions >= 0
    safe_positions = np.maximum(positions, 0)
    age = target - idx.take(safe_positions)
    valid &= age <= METRICS_LAG_TOLERANCE

    current = values.to_numpy(dtype=float)
    baseline = values.to_numpy(dtype=float)[safe_positions]
    valid &= np.isfinite(current) & np.isfinite(baseline)
    if percent:
        valid &= baseline != 0
        change = np.full(len(values), np.nan)
        np.divide(current, baseline, out=change, where=valid)
        change[valid] -= 1.0
    else:
        change = current - baseline
        change[~valid] = np.nan
    return pd.Series(change, index=idx, name=values.name)


def price_features(k4: pd.DataFrame, k1: pd.DataFrame, kd: pd.DataFrame) -> pd.DataFrame:
    p = pd.DataFrame(index=k4.index)
    c = k4["close"]
    r1 = np.log(c).diff()
    for n in (1, 3, 6, 12, 42):
        p[f"ret_{n}"] = np.log(c).diff(n)
    p["vol_6"] = r1.rolling(6).std()
    p["vol_42"] = r1.rolling(42).std()
    atr = _atr(k4)
    p["atr_pct"] = atr / c
    p["range_pct"] = (k4["high"] - k4["low"]) / c
    p["close_pos"] = (c - k4["low"]) / (k4["high"] - k4["low"]).replace(0, np.nan)
    v = k4["volume"]
    p["vol_z_42"] = (v - v.rolling(42).mean()) / v.rolling(42).std()
    p["taker_buy_ratio"] = k4["taker_buy_base"] / v.replace(0, np.nan)
    p[f"{CTX_PREFIX}close"] = c
    p[f"{CTX_PREFIX}atr"] = atr
    p = _stamp_closed(p, PRIMARY)

    if not k1.empty:
        h = pd.DataFrame(index=k1.index)
        lc = np.log(k1["close"])
        h["h1_ret_1"] = lc.diff()
        h["h1_ret_6"] = lc.diff(6)
        h["h1_vol_24"] = lc.diff().rolling(24).std()
        h["h1_volume_z_24"] = (k1["volume"] - k1["volume"].rolling(24).mean()) / k1["volume"].rolling(24).std()
        p = _asof(p, _stamp_closed(h, pd.Timedelta(hours=1)))

    if not kd.empty:
        d = pd.DataFrame(index=kd.index)
        ld = np.log(kd["close"])
        d["d1_ret_1"] = ld.diff()
        d["d1_ret_7"] = ld.diff(7)
        d["d1_vol_20"] = ld.diff().rolling(20).std()
        d["d1_dist_sma20"] = kd["close"] / kd["close"].rolling(20).mean() - 1
        p = _asof(p, _stamp_closed(d, pd.Timedelta(days=1)))
    return p


def derivatives_features(base: pd.DataFrame, funding: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    out = base
    if not funding.empty:
        f = pd.DataFrame({"funding_rate": funding["rate"]})
        f["funding_mean_3"] = funding["rate"].rolling(3).mean()
        f["funding_mean_21"] = funding["rate"].rolling(21).mean()
        out = _asof(out, f)
    if not metrics.empty:
        m = metrics.sort_index()
        g = pd.DataFrame(index=m.index)
        oi = m["oi_value"].where(m["oi_value"] > 0)
        g["oi_value"] = oi
        g["oi_chg_1h"] = _time_lagged_change(oi, 12)
        g["oi_chg_4h"] = _time_lagged_change(oi, 48)
        g["oi_chg_24h"] = _time_lagged_change(oi, 288)
        for col in ("top_ls_accounts", "top_ls_positions", "global_ls", "taker_ls_vol"):
            g[col] = m[col]
        g["global_ls_chg_4h"] = _time_lagged_change(m["global_ls"], 48, percent=False)
        out = _asof(out, g, tolerance=METRICS_MAX_AGE)
    return out


def liquidation_features(base: pd.DataFrame, liq: pd.DataFrame) -> pd.DataFrame:
    """Use 4h aggregate totals and preserve direction only for side-labelled events."""
    if liq.empty:
        return base
    out = base
    unknown = liq[liq["side"] == "UNKNOWN"]
    known = liq[liq["side"].isin(["LONG", "SHORT"])]
    known_total = pd.Series(dtype=float)
    known_qty = pd.Series(dtype=float)
    if not known.empty:
        hourly = known.assign(hour=known.index.floor("1h")).pivot_table(
            index="hour", columns="side", values="usd_value", aggfunc="sum"
        ).reindex(columns=["LONG", "SHORT"])
        hourly = hourly.sort_index().asfreq("1h").fillna(0.0)
        g_side = pd.DataFrame(index=hourly.index)
        for h in (1, 4, 24):
            g_side[f"liq_long_{h}h"] = hourly["LONG"].rolling(h, min_periods=1).sum()
            g_side[f"liq_short_{h}h"] = hourly["SHORT"].rolling(h, min_periods=1).sum()
        g_side["liq_imbalance_24h"] = (
            (g_side["liq_long_24h"] - g_side["liq_short_24h"])
            / (g_side["liq_long_24h"] + g_side["liq_short_24h"]).replace(0, np.nan)
        )
        out = _asof(out, _stamp_closed(g_side, pd.Timedelta(hours=1)),
                    tolerance=pd.Timedelta(hours=1))
        known_total = hourly.sum(axis=1, min_count=1).resample(
            "4h", origin="epoch", closed="left", label="right"
        ).sum(min_count=1)
        known_qty = known["qty"].resample(
            "4h", origin="epoch", closed="left", label="right"
        ).sum(min_count=1)
    else:
        # Keep a stable model input schema so later live side-labelled events do
        # not introduce entirely new columns after the first training run.
        for horizon in (1, 4, 24):
            out[f"liq_long_{horizon}h"] = np.nan
            out[f"liq_short_{horizon}h"] = np.nan
        out["liq_imbalance_24h"] = np.nan

    if unknown.empty and known_total.empty:
        return out
    if unknown.empty:
        combined_total = known_total
    else:
        # API BRICKS timestamps aggregates at their 4h period close. Fill only
        # between the first and last downloaded bucket; live side-labelled events
        # supply totals outside that historical coverage.
        api_grid = pd.date_range(unknown.index.min(), unknown.index.max(), freq="4h")
        api_total = unknown["usd_value"].resample(
            "4h", origin="epoch", closed="right", label="right"
        ).sum(min_count=1).reindex(api_grid)
        api_qty = unknown["qty"].resample(
            "4h", origin="epoch", closed="right", label="right"
        ).sum(min_count=1).reindex(api_grid)
        combined_total = api_total.combine_first(known_total)
        combined_qty = api_qty.combine_first(known_qty)
    if unknown.empty:
        combined_qty = known_qty

    full_grid = pd.date_range(
        min(combined_total.index.min(), combined_qty.index.min()),
        max(combined_total.index.max(), combined_qty.index.max()),
        freq="4h",
    )
    combined_total = combined_total.reindex(full_grid)
    combined_qty = combined_qty.reindex(full_grid)
    g_total = pd.DataFrame(index=full_grid)
    g_total["liq_total_4h"] = combined_total
    g_total["liq_total_24h"] = combined_total.rolling(6, min_periods=6).sum()
    g_total["liq_qty_4h"] = combined_qty
    g_total["liq_qty_24h"] = combined_qty.rolling(6, min_periods=6).sum()
    return _asof(out, g_total, tolerance=pd.Timedelta(hours=4))


def onchain_features(base: pd.DataFrame, onchain: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    """onchain: columns metric,value with DatetimeIndex. Shifted by publication lag, then ffilled."""
    if onchain.empty:
        return base
    wide = onchain.pivot_table(index=onchain.index, columns="metric", values="value", aggfunc="last")
    wide = wide.sort_index()
    feats = wide.add_prefix("oc_")
    chg = wide.pct_change(7, fill_method=None).add_prefix("oc_chg7_")
    feats = feats.join(chg)
    feats.index = feats.index + pd.Timedelta(hours=lag_hours)
    return _asof(base, feats)


def build_features(
    k4: pd.DataFrame,
    k1: pd.DataFrame,
    kd: pd.DataFrame,
    funding: pd.DataFrame,
    metrics: pd.DataFrame,
    liquidations: pd.DataFrame,
    onchain: pd.DataFrame,
    groups: FeatureGroups,
    lag_hours: int = 24,
) -> pd.DataFrame:
    """Index = decision time (candle close). Columns prefixed 'ctx_' are not model inputs."""
    feats = price_features(k4, k1, kd)
    if groups.derivatives:
        feats = derivatives_features(feats, funding, metrics)
    if groups.liquidations:
        feats = liquidation_features(feats, liquidations)
    if groups.onchain:
        feats = onchain_features(feats, onchain, lag_hours)
    feats.index.name = "decision_time"
    return feats


def model_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if not c.startswith(CTX_PREFIX)]
