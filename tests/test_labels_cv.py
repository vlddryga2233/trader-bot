import numpy as np
import pandas as pd

from trader_bot.labeling.labels import make_labels, to_class
from trader_bot.models.cv import purged_walk_forward


def test_labels_and_horizon_nan_tail():
    idx = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
    feats = pd.DataFrame({"ctx_close": [100, 100, 100, 110, 100, 100, 90, 100, 100, 100.0],
                          "atr_pct": 0.01}, index=idx)
    lab = make_labels(feats, horizon=3, flat_atr_mult=1.0)
    assert lab["label"].iloc[0] == 1.0       # 100 -> 110 (+10% > 1.7% band)
    assert lab["label"].iloc[3] == -1.0      # 110 -> 90? idx3->idx6 close 90
    assert lab["label"].iloc[1] == 0.0 or not np.isnan(lab["label"].iloc[1])
    assert lab["label"].iloc[-3:].isna().all()
    assert lab["label_end"].iloc[0] == idx[0] + pd.Timedelta(hours=12)
    assert set(to_class(lab["label"].dropna()).unique()) <= {0, 1, 2}


def test_purged_cv_never_overlaps_test_labels():
    horizon, embargo = 6, 6
    splits = list(purged_walk_forward(1000, 4, horizon, embargo, min_train=400))
    assert len(splits) == 4
    for tr, te in splits:
        assert tr.max() + horizon + embargo <= te.min()   # label of last train row ends before test
        assert te.min() > tr.max()
    # test blocks are consecutive and cover the tail
    assert splits[-1][1].max() == 999
    for (_, a), (_, b) in zip(splits, splits[1:]):
        assert a.max() + 1 == b.min()
