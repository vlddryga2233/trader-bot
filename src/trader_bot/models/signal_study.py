"""Tune signal and reward/risk thresholds on validation data, then score a later holdout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sqlalchemy.engine import Engine

from trader_bot.backtest.engine import simulate
from trader_bot.backtest.metrics import compute_metrics, passes_gate
from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.labeling.labels import LABEL_TO_CLASS, make_labels
from trader_bot.models.train import build_dataset, walk_forward_predictions

log = structlog.get_logger()
CLASS_NAMES = ("short", "flat", "long")
CONFIDENCE_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.75)
REWARD_RISK_GRID = (0.50, 0.75, 1.00, 1.25, 1.50)
MIN_VALIDATION_TRADES = 20
MAX_VALIDATION_DRAWDOWN = 0.15
MIN_VALIDATION_SHARPE = 0.0
MIN_VALIDATION_PROFIT_FACTOR = 1.0


def _class_counts(values: pd.Series) -> dict[str, dict[str, float | int]]:
    classes = values.map(LABEL_TO_CLASS).value_counts().reindex(range(3), fill_value=0)
    total = int(classes.sum())
    return {
        name: {
            "count": int(classes.loc[i]),
            "fraction": float(classes.loc[i] / total) if total else 0.0,
        }
        for i, name in enumerate(CLASS_NAMES)
    }


def _prediction_summary(preds: pd.DataFrame) -> dict:
    if preds.empty:
        return {"rows": 0, "predicted_classes": {}, "top_probability_quantiles": {}}
    probabilities = preds[["p_short", "p_flat", "p_long"]]
    predicted = probabilities.to_numpy().argmax(axis=1)
    top_probability = probabilities.max(axis=1)
    directional = predicted != 1
    agree = np.where(predicted == 0, preds["pred_ret"] < 0, preds["pred_ret"] > 0)
    counts = pd.Series(predicted).value_counts().reindex(range(3), fill_value=0)
    return {
        "rows": len(preds),
        "predicted_classes": {
            name: {"count": int(counts.loc[i]), "fraction": float(counts.loc[i] / len(preds))}
            for i, name in enumerate(CLASS_NAMES)
        },
        "directional_top_class": int(directional.sum()),
        "directional_and_return_agree": int((directional & agree).sum()),
        "top_probability_quantiles": {
            str(q): float(top_probability.quantile(q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "directional_coverage_by_threshold": {
            str(threshold): int((directional & (top_probability >= threshold) & agree).sum())
            for threshold in CONFIDENCE_GRID
        },
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _select_validation_candidate(validation_results: list[dict]) -> dict | None:
    """Return the best candidate only when it clears minimum quality and sample gates."""
    eligible = [
        result for result in validation_results
        if result["portfolio"].get("n_trades", 0) >= MIN_VALIDATION_TRADES
        and abs(result["portfolio"].get("max_drawdown", 1.0)) <= MAX_VALIDATION_DRAWDOWN
        and result["portfolio"].get("sharpe", float("-inf")) > MIN_VALIDATION_SHARPE
        and result["portfolio"].get("profit_factor", 0.0) is not None
        and result["portfolio"].get("profit_factor", 0.0) > MIN_VALIDATION_PROFIT_FACTOR
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda result: (
            result["portfolio"].get("sharpe", 0.0),
            result["portfolio"].get("profit_factor", 0.0) or 0.0,
            result["portfolio"].get("total_return", 0.0),
        ),
    )


def _simulate_portfolio(
    prepared: dict[str, dict],
    cfg: StrategyConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    confidence: float,
    reward_risk: float,
) -> tuple[dict, dict]:
    curves: list[pd.Series] = []
    all_trades: list[pd.DataFrame] = []
    by_symbol: dict[str, dict] = {}
    share = cfg.backtest.initial_equity / len(cfg.symbols)
    risk = cfg.risk.model_copy(update={"min_reward_risk": reward_risk})
    for symbol in cfg.symbols:
        item = prepared[symbol]
        preds = item["preds"]
        # Drop entries that cannot finish their max holding horizon before the
        # split boundary. This keeps validation trades out of holdout prices and
        # prevents open positions from being omitted at either segment's end.
        tradable_end = end - cfg.labeling.horizon_candles * pd.Timedelta(hours=4)
        preds = preds.loc[(preds.index >= start) & (preds.index <= tradable_end)]
        if preds.empty:
            continue
        bars = item["bars"].loc[(item["bars"].index >= start) & (item["bars"].index < end)]
        atr = item["atr"].reindex(preds.index)
        diagnostics: dict = {}
        equity, trades = simulate(
            bars,
            preds,
            atr,
            item["funding"],
            risk,
            cfg.backtest,
            confidence,
            cfg.labeling.horizon_candles,
            share,
            diagnostics,
        )
        curves.append(equity.rename(symbol))
        if not trades.empty:
            all_trades.append(trades.assign(symbol=symbol))
        by_symbol[symbol] = {
            **compute_metrics(equity, trades),
            "signal_diagnostics": diagnostics,
        }

    if not curves:
        return {"n_trades": 0, "gate_passed": False}, {}
    portfolio = pd.concat(curves, axis=1).ffill().bfill().sum(axis=1)
    trades_df = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades else pd.DataFrame(columns=["pnl"])
    )
    metrics = compute_metrics(portfolio, trades_df)
    metrics["gate_passed"] = passes_gate(metrics)
    return metrics, by_symbol


def run_signal_study(
    engine: Engine,
    cfg: StrategyConfig,
    holdout_start: pd.Timestamp,
    out_dir: Path,
) -> dict:
    """Tune on purged validation predictions and evaluate the chosen setting once on holdout."""
    start = pd.Timestamp(cfg.backtest.start, tz="UTC")
    holdout_start = pd.Timestamp(holdout_start).tz_convert("UTC")
    if holdout_start <= start:
        raise ValueError("holdout_start must be after backtest.start")

    prepared: dict[str, dict] = {}
    distributions: dict[str, dict] = {}
    for symbol in cfg.symbols:
        feats, raw = features_for_symbol(engine, symbol, cfg)
        ds = build_dataset(feats, cfg)
        preds = walk_forward_predictions(ds, cfg, None, start)
        labels = make_labels(ds.X, cfg.labeling.horizon_candles, cfg.labeling.flat_atr_mult)
        valid_labels = labels[labels["label"].notna()]
        # Purge validation labels whose forward outcome reaches into the final holdout.
        val_labels = valid_labels.loc[
            (valid_labels.index >= start)
            & (valid_labels["label_end"] < holdout_start)
        ]
        val_preds = preds.loc[(preds.index >= start) & (preds.index < holdout_start)]
        prepared[symbol] = {
            "preds": preds,
            "bars": raw["k4"],
            "atr": ds.X["ctx_atr"],
            "funding": raw["funding"]["rate"],
            "labels": valid_labels,
        }
        distributions[symbol] = {
            "pre_holdout_label_balance": _class_counts(
                valid_labels.loc[valid_labels["label_end"] < holdout_start, "label"]
            ),
            "validation_label_balance": _class_counts(val_labels["label"]),
            "validation_predictions": _prediction_summary(val_preds),
        }

    # The parameter grid and selection rule use validation results only.
    validation_results = []
    validation_end = holdout_start
    for confidence in CONFIDENCE_GRID:
        for reward_risk in REWARD_RISK_GRID:
            metrics, per_symbol = _simulate_portfolio(
                prepared, cfg, start, validation_end, confidence, reward_risk
            )
            validation_results.append({
                "confidence_threshold": confidence,
                "min_reward_risk": reward_risk,
                "portfolio": metrics,
                "per_symbol": per_symbol,
            })

    selected = _select_validation_candidate(validation_results)
    if selected is None:
        # Do not promote a weak validation winner based only on its trade count.
        default_metrics, default_by_symbol = _simulate_portfolio(
            prepared,
            cfg,
            start,
            validation_end,
            cfg.model.confidence_threshold,
            cfg.risk.min_reward_risk,
        )
        selected = {
            "confidence_threshold": cfg.model.confidence_threshold,
            "min_reward_risk": cfg.risk.min_reward_risk,
            "portfolio": default_metrics,
            "per_symbol": default_by_symbol,
        }
        selection_status = "no_viable_validation_setting; default_thresholds"
    else:
        selection_status = "minimum_trade_count_and_quality"

    confidence = selected["confidence_threshold"]
    reward_risk = selected["min_reward_risk"]
    holdout_end = max(item["bars"].index.max() for item in prepared.values()) + pd.Timedelta(hours=4)
    for symbol, item in prepared.items():
        labels = item["labels"]
        holdout_labels = labels.loc[
            (labels.index >= holdout_start) & (labels["label_end"] <= holdout_end)
        ]
        preds = item["preds"].loc[item["preds"].index >= holdout_start]
        distributions[symbol]["holdout_label_balance_after_selection"] = _class_counts(
            holdout_labels["label"]
        )
        distributions[symbol]["holdout_predictions_after_selection"] = _prediction_summary(preds)

    holdout_metrics, holdout_per_symbol = _simulate_portfolio(
        prepared, cfg, holdout_start, holdout_end, confidence, reward_risk
    )
    report = {
        "method": {
            "validation_start": str(start),
            "validation_end_exclusive": str(holdout_start),
            "holdout_start": str(holdout_start),
            "holdout_end_exclusive": str(holdout_end),
            "holdout_predictions_are_walk_forward": True,
            "validation_label_purge": "label_end must be before holdout_start",
            "confidence_grid": CONFIDENCE_GRID,
            "min_reward_risk_grid": REWARD_RISK_GRID,
            "selection_rule": (
                f"maximize validation Sharpe only with at least {MIN_VALIDATION_TRADES} trades, "
                f"positive Sharpe, profit factor > {MIN_VALIDATION_PROFIT_FACTOR:.2f}, and "
                f"max drawdown <= {MAX_VALIDATION_DRAWDOWN:.0%}; otherwise report defaults "
                "as unvalidated"
            ),
        },
        "class_and_probability_diagnostics": distributions,
        "selected_parameters": {
            "confidence_threshold": confidence,
            "min_reward_risk": reward_risk,
            "selection_status": selection_status,
            "validation_metrics": selected["portfolio"],
            "validation_by_symbol": selected["per_symbol"],
        },
        "validation_grid": validation_results,
        "final_holdout": {
            "portfolio": holdout_metrics,
            "per_symbol": holdout_per_symbol,
        },
        "active_config_unchanged": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(_json_safe(report), indent=2, default=str, allow_nan=False)
    )
    log.info(
        "signal_study.done",
        output=str(out_dir / "report.json"),
        confidence_threshold=confidence,
        min_reward_risk=reward_risk,
        validation_trades=selected["portfolio"].get("n_trades"),
        holdout_trades=holdout_metrics.get("n_trades"),
    )
    return report
