"""Triple-barrier labels matching the traded stop, target, and time-exit outcome.

The decision is made at row i's close. The path is rows i+1..i+horizon, and the stop is
checked before the take-profit within a candle, matching the backtester.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_STEP = pd.Timedelta(hours=4)


def make_barrier_labels(
    feats: pd.DataFrame,
    horizon: int,
    stop_mult: float,
    tp_mult: float,
    step: pd.Timedelta = DEFAULT_STEP,
) -> pd.DataFrame:
    """Return realized R, TP-first targets, and the timestamp where each label ends."""
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


def limit_entry_outcomes(feats: pd.DataFrame, horizon: int, stop_mult: float, tp_mult: float,
                         offset_atr: float) -> pd.DataFrame:
    """Compute R for one-candle limit entries; unfilled rows have zero return."""
    close = feats["ctx_close"].to_numpy(dtype=float)
    high = feats["ctx_high"].to_numpy(dtype=float)
    low = feats["ctx_low"].to_numpy(dtype=float)
    atr = feats["ctx_atr"].to_numpy(dtype=float)
    n = len(feats)
    stop_d, tp_d = stop_mult * atr, tp_mult * atr
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
                stop_hit = low[i + j] <= stop if side == 1 else high[i + j] >= stop
                if stop_hit:
                    result = -1.0
                    break
                tp_hit = high[i + j] >= tp if side == 1 else low[i + j] <= tp
                if j > 1 and tp_hit:
                    result = rr
                    break
            if result is None:
                result = side * (close[i + horizon] - entry[i]) / stop_d[i]
            r[i] = result
        out[f"r_{name}_limit"] = r
        out[f"filled_{name}"] = filled
    return out
