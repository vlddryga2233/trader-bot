"""Simulate a multi-symbol portfolio over a date segment with any signal function."""
from __future__ import annotations

import pandas as pd

from trader_bot.backtest.engine import BAR, SignalFn, simulate
from trader_bot.backtest.metrics import bootstrap_sharpe_ci, compute_metrics, passes_gate
from trader_bot.config.settings import StrategyConfig


def simulate_segment(prepared: dict[str, dict], cfg: StrategyConfig, seg_start: pd.Timestamp,
                     seg_end: pd.Timestamp, signal_fn: SignalFn) -> dict:
    """Simulate prepared symbol inputs, dropping entries that cannot finish before seg_end."""
    share = cfg.backtest.initial_equity / len(cfg.symbols)
    tradable_end = seg_end - cfg.labeling.horizon_candles * BAR
    curves, trades_all, per_symbol = [], [], {}
    for symbol, item in prepared.items():
        preds = item["preds"]
        preds = preds.loc[(preds.index >= seg_start) & (preds.index <= tradable_end)]
        if preds.empty:
            continue
        bars = item["bars"].loc[(item["bars"].index >= seg_start) & (item["bars"].index < seg_end)]
        diagnostics: dict = {}
        equity, trades = simulate(
            bars, preds, item["atr"].reindex(preds.index), item["funding"], cfg.risk,
            cfg.backtest, 0.0, cfg.labeling.horizon_candles, share, diagnostics,
            signal_fn=signal_fn,
        )
        curves.append(equity.rename(symbol))
        if not trades.empty:
            trades_all.append(trades.assign(symbol=symbol))
        per_symbol[symbol] = {**compute_metrics(equity, trades),
                              "signal_diagnostics": diagnostics}
    if not curves:
        return {"portfolio": {"n_trades": 0, "gate_passed": False}, "per_symbol": {}}
    portfolio = pd.concat(curves, axis=1).ffill().bfill().sum(axis=1)
    trades_df = (pd.concat(trades_all, ignore_index=True) if trades_all
                 else pd.DataFrame(columns=["pnl"]))
    metrics = compute_metrics(portfolio, trades_df)
    metrics["gate_passed"] = passes_gate(metrics)
    metrics["sharpe_ci_90"] = bootstrap_sharpe_ci(portfolio)
    return {"portfolio": metrics, "per_symbol": per_symbol}


def segments_for(start: pd.Timestamp, end: pd.Timestamp,
                 holdout_start: pd.Timestamp | None) -> dict[str, tuple]:
    if holdout_start is None:
        return {"all": (start, end)}
    return {"validation": (start, holdout_start), "holdout": (holdout_start, end)}
