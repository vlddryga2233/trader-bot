from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def purged_walk_forward(
    n: int, n_splits: int, horizon: int, embargo: int, min_train: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward splits over n time-ordered samples.

    The data after `min_train` is cut into n_splits consecutive test blocks. Training indices
    stop `horizon + embargo` samples before the test block starts, so no training label
    (which looks `horizon` candles ahead) overlaps the test period.
    """
    if n <= min_train + n_splits:
        raise ValueError(f"not enough samples ({n}) for min_train={min_train}, splits={n_splits}")
    block = (n - min_train) // n_splits
    for i in range(n_splits):
        test_start = min_train + i * block
        test_end = n if i == n_splits - 1 else test_start + block
        train_end = test_start - horizon - embargo
        if train_end <= 0:
            continue
        yield np.arange(0, train_end), np.arange(test_start, test_end)
