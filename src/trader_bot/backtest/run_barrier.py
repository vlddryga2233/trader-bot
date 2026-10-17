from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy.engine import Engine

from trader_bot.backtest.engine import BAR
from trader_bot.backtest.segments import segments_for, simulate_segment
from trader_bot.config.settings import StrategyConfig
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.models.barrier import build_barrier_dataset, walk_forward_barrier_predictions
from trader_bot.risk.engine import make_ev_signal, round_trip_cost

log = structlog.get_logger()


def ev_signal_fn(cfg: StrategyConfig):
    cost = round_trip_cost(cfg.backtest.taker_fee, cfg.backtest.slippage_bps)

    def fn(row: pd.Series, price: float, atr: float):
        return make_ev_signal(float(row["p_long_win"]), float(row["p_short_win"]), price, atr,
                              cfg.risk.atr_stop_mult, cfg.barrier.tp_atr_mult, cost,
                              cfg.barrier.min_ev_r)
    return fn


def run_barrier_backtest(engine: Engine, cfg: StrategyConfig, out_dir: Path,
                         holdout_start: pd.Timestamp | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(cfg.backtest.start, tz="UTC")
    prepared = {}
    for symbol in cfg.symbols:
        feats, raw = features_for_symbol(engine, symbol, cfg)
        ds = build_barrier_dataset(feats, cfg)
        preds = walk_forward_barrier_predictions(ds, cfg, start)
        if preds.empty:
            log.warning("barrier_backtest.no_predictions", symbol=symbol)
            continue
        preds.to_csv(out_dir / f"{symbol}_barrier_predictions.csv")
        prepared[symbol] = {"preds": preds, "bars": raw["k4"], "atr": ds.X["ctx_atr"],
                            "funding": raw["funding"]["rate"]}
    if not prepared:
        raise RuntimeError("no barrier predictions; is there enough history in the DB?")
    end = max(item["bars"].index.max() for item in prepared.values()) + BAR
    signal_fn = ev_signal_fn(cfg)
    report = {
        "settings": {
            "signal_mode": "barrier",
            "horizon_candles": cfg.labeling.horizon_candles,
            "atr_stop_mult": cfg.risk.atr_stop_mult,
            "tp_atr_mult": cfg.barrier.tp_atr_mult,
            "min_ev_r": cfg.barrier.min_ev_r,
            "min_reward_risk": cfg.risk.min_reward_risk,
            "taker_fee": cfg.backtest.taker_fee,
            "slippage_bps": cfg.backtest.slippage_bps,
            "holdout_start": str(holdout_start) if holdout_start is not None else None,
        },
        "segments": {
            name: simulate_segment(prepared, cfg, a, b, signal_fn)
            for name, (a, b) in segments_for(start, end, holdout_start).items()
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
