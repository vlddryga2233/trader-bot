"""Pool several symbols' barrier datasets into one time-ordered training set."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader_bot.config.settings import StrategyConfig
from trader_bot.models.barrier import (
    FOLD_KEYS,
    BarrierDataset,
    BarrierModel,
    barrier_signal_metrics,
    summarize_folds,
)
from trader_bot.models.cv import purged_walk_forward

SYMBOL_COL = "sym_id"


def pool_datasets(datasets: dict[str, BarrierDataset]) -> tuple[BarrierDataset, pd.Series]:
    """Return the pooled labeled dataset (sorted by time, then symbol) and a symbol Series."""
    frames, symbols = [], list(datasets)
    for code, symbol in enumerate(symbols):
        ds = datasets[symbol]
        lab = ds.has_label.to_numpy(dtype=bool)
        frame = ds.X[lab].copy()
        frame[SYMBOL_COL] = code
        frame["_y_long"] = ds.y_long[lab].to_numpy()
        frame["_y_short"] = ds.y_short[lab].to_numpy()
        frame["_r_long"] = ds.r_long[lab].to_numpy()
        frame["_r_short"] = ds.r_short[lab].to_numpy()
        frame["_symbol"] = symbol
        frames.append(frame.reset_index())
    time_col = frames[0].columns[0]
    pooled = pd.concat(frames, ignore_index=True).sort_values([time_col, SYMBOL_COL])
    pooled = pooled.set_index(time_col)
    columns = sorted(set.intersection(*(set(datasets[s].columns) for s in symbols)))
    columns.append(SYMBOL_COL)
    out = BarrierDataset(
        pooled.drop(columns=["_y_long", "_y_short", "_r_long", "_r_short", "_symbol"]),
        pooled["_y_long"], pooled["_y_short"], pooled["_r_long"], pooled["_r_short"],
        pd.Series(True, index=pooled.index), columns,
    )
    return out, pooled["_symbol"]


def cv_evaluate_pooled(datasets: dict[str, BarrierDataset], cfg: StrategyConfig) -> dict:
    """Purged CV on pooled rows; metrics are reported per symbol from pooled OOS predictions.

    Rows are interleaved (one per symbol per timestamp), so purge/embargo/min_train are scaled
    by the number of symbols to keep the same time gaps as single-symbol CV.
    """
    ds, symbols = pool_datasets(datasets)
    sym = symbols.to_numpy()
    k = len(datasets)
    m, h = cfg.model, cfg.labeling.horizon_candles
    X = ds.X

    def score(P: pd.DataFrame, rows: np.ndarray, symbol: str) -> dict:
        mask = sym[rows] == symbol
        r = rows[mask]
        return barrier_signal_metrics(
            P.iloc[np.flatnonzero(mask)], X.iloc[r], ds.y_long.iloc[r], ds.y_short.iloc[r],
            ds.r_long.iloc[r], ds.r_short.iloc[r], cfg,
        )

    preds, all_rows = [], []
    folds: dict[str, list[dict]] = {symbol: [] for symbol in datasets}
    for fold, (tr, te) in enumerate(
        purged_walk_forward(len(X), m.cv_splits, h * k, m.embargo_candles * k,
                            m.min_train_candles * k),
        start=1,
    ):
        model = BarrierModel(ds.columns, purge=(h + m.embargo_candles) * k,
                             calib_frac=cfg.barrier.calib_frac).fit(
            X.iloc[tr], ds.y_long.iloc[tr], ds.y_short.iloc[tr]
        )
        P = model.predict(X.iloc[te])
        preds.append(P)
        all_rows.append(te)
        for symbol in datasets:
            fm = score(P, te, symbol)
            folds[symbol].append({"fold": fold, "n_test": fm["n_oos"],
                                  **{key: fm[key] for key in FOLD_KEYS}})
    P, rows = pd.concat(preds), np.concatenate(all_rows)
    return {symbol: summarize_folds(score(P, rows, symbol), folds[symbol])
            for symbol in datasets}
