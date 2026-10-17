"""Daily decision features for research. Index values are daily candle close times."""

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
        f = pd.DataFrame(
            {
                "funding_mean_3": rate.rolling(3).mean(),
                "funding_mean_21": rate.rolling(21).mean(),
                "funding_z_90": (rate - rate.rolling(90).mean()) / rate.rolling(90).std(),
            }
        )
        p = _asof(p, f)
    p.index.name = "decision_time"
    return p
