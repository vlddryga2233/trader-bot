"""Meta-labeling: a rule picks the side; a calibrated classifier decides whether to trade."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from trader_bot.backtest.baselines import baseline_direction
from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.models.barrier import (
    FOLD_KEYS,
    BarrierModel,
    barrier_signal_metrics,
    build_barrier_dataset,
    summarize_folds,
)
from trader_bot.models.cv import purged_walk_forward
from trader_bot.models.selection import select

PRIMARIES = ("breakout_20", "momentum_42")


def _as_barrier_preds(p: np.ndarray, d: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Put probability on the side chosen by the primary rule."""
    return pd.DataFrame(
        {"p_long_win": np.where(d == 1, p, 0.0), "p_short_win": np.where(d == -1, p, 0.0)},
        index=index,
    )


def cv_evaluate_meta(feats: pd.DataFrame, cfg: StrategyConfig, primary: str) -> dict:
    """Purged CV where each fold trains only on training rows where the rule fires."""
    ds = build_barrier_dataset(feats, cfg)
    lab = ds.labeled
    X = ds.X.iloc[lab]
    d = baseline_direction(feats, primary).reindex(X.index).fillna(0).astype(int).to_numpy()
    yl, ys = ds.y_long.iloc[lab], ds.y_short.iloc[lab]
    rl, rs = ds.r_long.iloc[lab], ds.r_short.iloc[lab]
    y_rule = pd.Series(np.where(d == 1, yl, ys), index=X.index)
    m, h = cfg.model, cfg.labeling.horizon_candles
    preds, folds = [], []
    for fold, (tr, te) in enumerate(
        purged_walk_forward(len(X), m.cv_splits, h, m.embargo_candles, m.min_train_candles),
        start=1,
    ):
        fire = tr[d[tr] != 0]
        model = BarrierModel(
            ds.columns,
            purge=h + m.embargo_candles,
            calib_frac=cfg.barrier.calib_frac,
            size=cfg.barrier.model_size,
        ).fit(X.iloc[fire], y_rule.iloc[fire], None)
        p = model.predict(X.iloc[te])["p_long_win"].to_numpy()
        P = _as_barrier_preds(p, d[te], X.index[te])
        preds.append(P)
        fm = barrier_signal_metrics(
            P, X.iloc[te], yl.iloc[te], ys.iloc[te], rl.iloc[te], rs.iloc[te], cfg, primary=d[te]
        )
        folds.append(
            {
                "fold": fold,
                "n_train_rule_rows": len(fire),
                "n_test": len(te),
                **{key: fm[key] for key in FOLD_KEYS if key in fm},
            }
        )
    P = pd.concat(preds)
    pos = X.index.get_indexer(P.index)
    metrics = barrier_signal_metrics(
        P, X.iloc[pos], yl.iloc[pos], ys.iloc[pos], rl.iloc[pos], rs.iloc[pos], cfg, primary=d[pos]
    )
    metrics["n_rule_rows"] = int((d != 0).sum())
    return summarize_folds(metrics, folds)


def run_meta_experiment(
    engine: Engine,
    cfg: StrategyConfig,
    out_dir: Path,
    holdout_start: pd.Timestamp,
    primaries: list[str],
    horizons: list[int],
    tp_atr_mults: list[float],
    t_min: float = 3.0,
    min_trades: int = 40,
) -> dict:
    report: dict = {
        "holdout_start": str(holdout_start),
        "mode": "meta",
        "symbols": {},
        "settings": {
            "model_size": cfg.barrier.model_size,
            "taker_fee": cfg.backtest.taker_fee,
            "slippage_bps": cfg.backtest.slippage_bps,
        },
    }
    for symbol in cfg.symbols:
        feats, _ = features_for_symbol(engine, symbol, cfg)
        feats = feats.loc[feats.index < holdout_start]
        results = []
        for primary in primaries:
            for horizon in horizons:
                for tp in tp_atr_mults:
                    variant = cfg.model_copy(
                        update={
                            "labeling": cfg.labeling.model_copy(
                                update={"horizon_candles": horizon}
                            ),
                            "barrier": cfg.barrier.model_copy(update={"tp_atr_mult": tp}),
                        }
                    )
                    results.append(
                        {
                            "primary": primary,
                            "horizon_candles": horizon,
                            "tp_atr_mult": tp,
                            "metrics": cv_evaluate_meta(feats, variant, primary),
                        }
                    )
        report["symbols"][symbol] = results
    report["selection"] = select(report, "meta", t_min, min_trades)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
