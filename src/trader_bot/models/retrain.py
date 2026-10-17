from __future__ import annotations

import structlog
from sqlalchemy.engine import Engine

from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.models.registry import ModelRegistry
from trader_bot.models.train import build_dataset, cv_evaluate, fit_final, tune

log = structlog.get_logger()


def retrain_symbol(engine: Engine, cfg: StrategyConfig, registry: ModelRegistry, symbol: str,
                   do_tune: bool = False, promote_model: bool = True) -> dict:
    """Train on labeled history and promote only if CV directional quality clears its gates."""
    feats, _ = features_for_symbol(engine, symbol, cfg)
    ds = build_dataset(feats, cfg)
    params = tune(ds, cfg) if do_tune else registry.current_params(symbol)
    metrics = cv_evaluate(ds, cfg, params)
    ens = fit_final(ds, params, cfg.model.class_balance)
    version = registry.save(symbol, ens, metrics, {
        "params": params,
        "n_train": len(ds.labeled),
        "flat_atr_mult": cfg.labeling.flat_atr_mult,
        "horizon_candles": cfg.labeling.horizon_candles,
        "class_balance": cfg.model.class_balance,
    })
    promoted = promote_model and registry.should_promote(
        symbol,
        metrics,
        min_balanced_accuracy=cfg.model.promotion_min_balanced_accuracy,
        min_signal_precision=cfg.model.promotion_min_signal_precision,
        min_signal_coverage=cfg.model.promotion_min_signal_coverage,
    )
    if promoted:
        registry.promote(symbol, version)
    log.info("retrain.done", symbol=symbol, version=version, promoted=promoted, **{
        k: v for k, v in metrics.items() if v is not None})
    return {"version": version, "promoted": promoted, "metrics": metrics}
