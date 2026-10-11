from __future__ import annotations

import json
from pathlib import Path

import structlog
from sqlalchemy.engine import Engine

from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.models.train import build_dataset, cv_evaluate

log = structlog.get_logger()


def compare_label_and_weight_variants(
    engine: Engine,
    cfg: StrategyConfig,
    out_dir: Path,
    flat_atr_mults: list[float] | None = None,
) -> dict:
    """Compare label bands and balanced classifiers using identical purged CV folds."""
    multipliers = flat_atr_mults or [0.5, 1.0, 1.5]
    if any(mult <= 0 for mult in multipliers):
        raise ValueError("all flat ATR multipliers must be positive")

    report = {"symbols": {}, "variants": []}
    for symbol in cfg.symbols:
        feats, _ = features_for_symbol(engine, symbol, cfg)
        symbol_results = []
        for multiplier in multipliers:
            for class_balance in (False, True):
                variant_cfg = cfg.model_copy(update={
                    "labeling": cfg.labeling.model_copy(
                        update={"flat_atr_mult": multiplier}
                    ),
                    "model": cfg.model.model_copy(update={"class_balance": class_balance}),
                })
                ds = build_dataset(feats, variant_cfg)
                metrics = cv_evaluate(ds, variant_cfg)
                result = {
                    "flat_atr_mult": multiplier,
                    "horizon_candles": cfg.labeling.horizon_candles,
                    "class_balance": class_balance,
                    "metrics": metrics,
                }
                symbol_results.append(result)
                log.info(
                    "experiment.variant.done", symbol=symbol, flat_atr_mult=multiplier,
                    class_balance=class_balance,
                    balanced_accuracy=metrics["balanced_accuracy"],
                    directional_hc_coverage=metrics["directional_hc_coverage"],
                )
        report["symbols"][symbol] = symbol_results

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
