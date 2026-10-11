from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy.engine import Engine

from trader_bot.backtest.engine import simulate
from trader_bot.backtest.metrics import compute_metrics, passes_gate
from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.models.train import build_dataset, walk_forward_predictions

log = structlog.get_logger()


def run_backtest(engine: Engine, cfg: StrategyConfig, out_dir: Path,
                 params_by_symbol: dict[str, dict] | None = None) -> dict:
    """Walk-forward OOS backtest. Capital is split equally across symbols (independent books)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(cfg.backtest.start, tz="UTC")
    share = cfg.backtest.initial_equity / len(cfg.symbols)
    curves, all_trades, per_symbol, diagnostics_by_symbol = [], [], {}, {}

    for symbol in cfg.symbols:
        feats, raw = features_for_symbol(engine, symbol, cfg)
        ds = build_dataset(feats, cfg)
        preds = walk_forward_predictions(ds, cfg, (params_by_symbol or {}).get(symbol), start)
        if preds.empty:
            log.warning("backtest.no_predictions", symbol=symbol)
            continue
        atr = ds.X.loc[preds.index, "ctx_atr"]
        diagnostics = {}
        equity, trades = simulate(
            raw["k4"], preds, atr, raw["funding"]["rate"], cfg.risk, cfg.backtest,
            cfg.model.confidence_threshold, cfg.labeling.horizon_candles, share, diagnostics,
        )
        curves.append(equity.rename(symbol))
        if len(trades):
            all_trades.append(trades.assign(symbol=symbol))
        per_symbol[symbol] = compute_metrics(equity, trades)
        diagnostics_by_symbol[symbol] = diagnostics
        preds.to_csv(out_dir / f"{symbol}_predictions.csv")

    if not curves:
        raise RuntimeError("no backtest results; is there enough history in the DB?")
    portfolio = pd.concat(curves, axis=1).ffill().bfill().sum(axis=1)
    trades_df = pd.concat(all_trades) if all_trades else pd.DataFrame(columns=["pnl"])
    metrics = compute_metrics(portfolio, trades_df)
    metrics["gate_passed"] = passes_gate(metrics)
    report = {
        "portfolio": metrics,
        "per_symbol": per_symbol,
        "diagnostics_by_symbol": diagnostics_by_symbol,
        "settings": {
            "flat_atr_mult": cfg.labeling.flat_atr_mult,
            "horizon_candles": cfg.labeling.horizon_candles,
            "confidence_threshold": cfg.model.confidence_threshold,
            "class_balance": cfg.model.class_balance,
            "min_reward_risk": cfg.risk.min_reward_risk,
            "atr_stop_mult": cfg.risk.atr_stop_mult,
            "taker_fee": cfg.backtest.taker_fee,
            "slippage_bps": cfg.backtest.slippage_bps,
        },
    }
    portfolio.to_csv(out_dir / "equity.csv", header=["equity"])
    trades_df.to_csv(out_dir / "trades.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
