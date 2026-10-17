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
