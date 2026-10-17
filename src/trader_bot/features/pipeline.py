"""Single entry point that turns DB contents into model features (used by train, backtest and live)."""
from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine

from trader_bot.config.settings import StrategyConfig
from trader_bot.db import schema, store
from trader_bot.features.builder import build_features


def load_raw(engine: Engine, symbol: str) -> dict[str, pd.DataFrame]:
    def k(interval: str) -> pd.DataFrame:
        return store.load_frame(engine, schema.klines, "open_time", symbol,
                                extra={"interval": interval})

    onchain = store.load_frame(engine, schema.onchain, "ts", None, extra={"asset": symbol[:-4]})
    return {
        "k4": k("4h"), "k1": k("1h"), "kd": k("1d"),
        "funding": store.load_frame(engine, schema.funding, "ts", symbol),
        "metrics": store.load_frame(engine, schema.oi_metrics, "ts", symbol),
        "liquidations": store.load_frame(engine, schema.liquidations, "ts", symbol),
        "onchain": onchain,
    }


def reference_symbol(symbol: str) -> str:
    return "ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT"


def features_for_symbol(engine: Engine, symbol: str, cfg: StrategyConfig) -> tuple[pd.DataFrame, dict]:
    raw = load_raw(engine, symbol)
    ref_k4 = None
    if cfg.features.groups.cross_asset:
        ref_k4 = store.load_frame(engine, schema.klines, "open_time", reference_symbol(symbol),
                                  extra={"interval": "4h"})
    feats = build_features(
        raw["k4"], raw["k1"], raw["kd"], raw["funding"], raw["metrics"],
        raw["liquidations"], raw["onchain"], cfg.features.groups,
        cfg.features.onchain_publication_lag_hours, ref_k4=ref_k4,
    )
    return feats, raw
