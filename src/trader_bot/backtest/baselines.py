"""Simple trailing-only strategies evaluated with the same simulator, sizing and costs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from trader_bot.backtest.engine import BAR
from trader_bot.backtest.segments import segments_for, simulate_segment
from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.risk.engine import Signal

BASELINES = ("always_long", "momentum_42", "breakout_20", "funding_contrarian")


def baseline_direction(feats: pd.DataFrame, name: str) -> pd.Series:
    """Return +1/-1/0 per decision time, using only data available at that time."""
    close = feats["ctx_close"]
    if name == "always_long":
        return pd.Series(1, index=feats.index)
    if name == "momentum_42":
        return np.sign(feats["ret_42"]).fillna(0).astype(int)
    if name == "breakout_20":
        upper = feats["ctx_high"].rolling(20).max().shift(1)
        lower = feats["ctx_low"].rolling(20).min().shift(1)
        return pd.Series(np.where(close > upper, 1, np.where(close < lower, -1, 0)),
                         index=feats.index)
    if name == "funding_contrarian":
        if "funding_rate" not in feats:
            return pd.Series(0, index=feats.index)
        funding = feats["funding_rate"]
        z = (funding - funding.rolling(180).mean()) / funding.rolling(180).std()
        return pd.Series(np.where(z > 2, -1, np.where(z < -2, 1, 0)), index=feats.index)
    raise ValueError(f"unknown baseline {name}")


def baseline_signal_fn(cfg: StrategyConfig):
    """Place the target exactly at the configured minimum reward/risk."""
    tp_mult = cfg.risk.min_reward_risk * cfg.risk.atr_stop_mult

    def fn(row: pd.Series, price: float, atr: float):
        direction = int(row["direction"])
        if direction == 0 or price <= 0 or atr <= 0:
            return None
        return Signal(direction, 1.0, direction * tp_mult * atr / price * (1 + 1e-9))
    return fn


def run_baselines(engine: Engine, cfg: StrategyConfig, out_dir: Path,
                  holdout_start: pd.Timestamp | None = None) -> dict:
    start = pd.Timestamp(cfg.backtest.start, tz="UTC")
    data = {}
    for symbol in cfg.symbols:
        feats, raw = features_for_symbol(engine, symbol, cfg)
        data[symbol] = (feats, raw)
    end = max(raw["k4"].index.max() for _, raw in data.values()) + BAR
    report: dict = {"baselines": {}}
    for name in BASELINES:
        prepared = {}
        for symbol, (feats, raw) in data.items():
            direction = baseline_direction(feats, name)
            preds = pd.DataFrame({"direction": direction}).loc[lambda frame: frame.index >= start]
            prepared[symbol] = {"preds": preds, "bars": raw["k4"], "atr": feats["ctx_atr"],
                                "funding": raw["funding"]["rate"]}
        report["baselines"][name] = {
            segment: simulate_segment(prepared, cfg, a, b, baseline_signal_fn(cfg))
            for segment, (a, b) in segments_for(start, end, holdout_start).items()
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
