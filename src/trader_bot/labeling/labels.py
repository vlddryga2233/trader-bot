from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}  # short, flat, long


def make_labels(feats: pd.DataFrame, horizon: int, flat_atr_mult: float,
                step: pd.Timedelta = pd.Timedelta(hours=4)) -> pd.DataFrame:
    """Targets for a decision taken at the close of each candle.

    fwd_ret   regression target: close[t+h] / close[t] - 1
    label     -1 / 0 / +1 using a volatility-scaled flat band k * atr_pct * sqrt(h)
    label_end time at which the label is fully known (used for purging in CV)
    Rows without a complete horizon get NaN targets.
    """
    close = feats["ctx_close"]
    fwd = close.shift(-horizon) / close - 1
    band = flat_atr_mult * feats["atr_pct"] * np.sqrt(horizon)
    label = pd.Series(0.0, index=feats.index)
    label[fwd > band] = 1.0
    label[fwd < -band] = -1.0
    label[fwd.isna() | band.isna()] = np.nan
    out = pd.DataFrame({"fwd_ret": fwd, "label": label})
    out["label_end"] = feats.index + horizon * step
    return out


def to_class(label: pd.Series) -> pd.Series:
    return label.map(LABEL_TO_CLASS).astype(int)
