from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy.engine import Engine

from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.models.barrier import build_barrier_dataset, cv_evaluate_barrier
from trader_bot.models.pooled import cv_evaluate_pooled
from trader_bot.models.selection import select

log = structlog.get_logger()
SUMMARY_KEYS = ("reward_risk", "rr_meets_risk_floor", "auc_long", "auc_short", "n_signals",
                "signal_coverage", "hit_rate_on_signals", "mean_net_r_on_signals",
                "folds_with_positive_mean_r", "always_long_mean_net_r",
                "n_trades_nonoverlap", "mean_net_r_nonoverlap", "t_stat_nonoverlap",
                "folds_with_positive_nonoverlap_r", "folds_with_signals")


def run_barrier_experiment(engine: Engine, cfg: StrategyConfig, out_dir: Path,
                           holdout_start: pd.Timestamp, tp_atr_mults: list[float],
                           horizons: list[int], pooled: bool = False,
                           t_min: float = 3.0, min_trades: int = 40) -> dict:
    """Grid barrier variants using only pre-holdout rows for model selection."""
    report: dict = {
        "holdout_start": str(holdout_start),
        "symbols": {},
        "pooled": pooled,
        "settings": {
            "feature_groups": cfg.features.groups.model_dump(),
            "model_size": cfg.barrier.model_size,
            "taker_fee": cfg.backtest.taker_fee,
            "slippage_bps": cfg.backtest.slippage_bps,
            "maker_fee": cfg.backtest.maker_fee,
            "limit_offset_atr": cfg.barrier.limit_offset_atr,
            "decision_interval": cfg.decision_interval,
            "min_train_candles": cfg.model.min_train_candles,
        },
    }
    if pooled:
        features = {}
        for symbol in cfg.symbols:
            feats, _ = features_for_symbol(engine, symbol, cfg)
            features[symbol] = feats.loc[feats.index < holdout_start]
        for horizon in horizons:
            for tp in tp_atr_mults:
                variant = cfg.model_copy(update={
                    "labeling": cfg.labeling.model_copy(update={"horizon_candles": horizon}),
                    "barrier": cfg.barrier.model_copy(update={"tp_atr_mult": tp}),
                })
                datasets = {symbol: build_barrier_dataset(feats, variant)
                            for symbol, feats in features.items()}
                metrics_by_symbol = cv_evaluate_pooled(datasets, variant)
                for symbol, metrics in metrics_by_symbol.items():
                    report["symbols"].setdefault(symbol, []).append({
                        "horizon_candles": horizon, "tp_atr_mult": tp, "metrics": metrics,
                    })
                    log.info("barrier_experiment.variant", symbol=symbol, horizon=horizon, tp=tp,
                             **{key: metrics[key] for key in SUMMARY_KEYS})
    else:
        for symbol in cfg.symbols:
            feats, _ = features_for_symbol(engine, symbol, cfg)
            feats = feats.loc[feats.index < holdout_start]
            results = []
            for horizon in horizons:
                for tp in tp_atr_mults:
                    variant = cfg.model_copy(update={
                        "labeling": cfg.labeling.model_copy(update={"horizon_candles": horizon}),
                        "barrier": cfg.barrier.model_copy(update={"tp_atr_mult": tp}),
                    })
                    metrics = cv_evaluate_barrier(build_barrier_dataset(feats, variant), variant)
                    results.append({"horizon_candles": horizon, "tp_atr_mult": tp,
                                    "metrics": metrics})
                    log.info("barrier_experiment.variant", symbol=symbol, horizon=horizon, tp=tp,
                             **{key: metrics[key] for key in SUMMARY_KEYS})
            report["symbols"][symbol] = results
    mode = "limit" if cfg.barrier.limit_offset_atr > 0 else "barrier"
    report["selection"] = select(report, mode, t_min, min_trades)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
