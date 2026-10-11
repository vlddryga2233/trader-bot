from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pandas as pd
import structlog
from sqlalchemy.engine import Engine

from trader_bot.config.settings import Settings, StrategyConfig
from trader_bot.data.ingest.backfill import backfill_symbol
from trader_bot.db import schema
from trader_bot.execution.client import ExchangeClient
from trader_bot.execution.manager import OrderManager, PositionBook
from trader_bot.execution.user_stream import AccountState, run_user_stream
from trader_bot.features.pipeline import features_for_symbol
from trader_bot.models.registry import ModelRegistry
from trader_bot.models.retrain import retrain_symbol
from trader_bot.risk.engine import Rejection, make_signal, plan_trade
from trader_bot.risk.kill_switch import KillSwitch

log = structlog.get_logger()
BAR = timedelta(hours=4)
GRACE = timedelta(seconds=20)


def next_close(now: datetime) -> datetime:
    """Next 4h boundary (UTC) strictly after now."""
    base = now.replace(minute=0, second=0, microsecond=0)
    base -= timedelta(hours=base.hour % 4)
    return base + BAR


def _log_signal(engine: Engine, symbol: str, version: str, sig, accepted: bool, reason: str) -> None:
    with engine.begin() as conn:
        conn.execute(schema.signals.insert().values(
            ts=datetime.now(UTC), symbol=symbol, model_version=version,
            direction=sig.direction if sig else 0, confidence=sig.confidence if sig else None,
            pred_return=sig.pred_ret if sig else None, accepted=accepted, reason=reason))


def decide_and_trade(engine: Engine, cfg: StrategyConfig, registry: ModelRegistry,
                     mgr: OrderManager, symbol: str, equity: float, open_positions: int,
                     expected_close: datetime) -> bool:
    feats, _ = features_for_symbol(engine, symbol, cfg)
    if feats.empty or feats.index[-1] < pd.Timestamp(expected_close):
        log.warning("decision.stale_data", symbol=symbol,
                    last=str(feats.index[-1]) if len(feats) else None)
        return False
    row = feats.iloc[[-1]]
    ens, version = registry.load_current(symbol)
    pred = ens.predict(row).iloc[0]
    sig = make_signal(pred["p_short"], pred["p_flat"], pred["p_long"], pred["pred_ret"],
                      cfg.model.confidence_threshold)
    if sig is None:
        _log_signal(engine, symbol, version, None, False, "no signal")
        return False
    plan = plan_trade(sig, equity, float(row["ctx_close"].iloc[0]), float(row["ctx_atr"].iloc[0]),
                      cfg.risk, open_positions)
    if isinstance(plan, Rejection):
        _log_signal(engine, symbol, version, sig, False, plan.reason)
        return False
    ok = mgr.open_trade(symbol, plan)
    _log_signal(engine, symbol, version, sig, ok, "opened" if ok else "order failed")
    return ok


async def run_live(cfg: StrategyConfig, settings: Settings, engine: Engine, client: ExchangeClient) -> None:
    ks = KillSwitch(settings.state_dir, cfg.risk.daily_loss_limit, cfg.risk.weekly_loss_limit)
    if ks.is_locked():
        log.critical("kill_switch.locked_refusing_to_start", info=ks.lock_info())
        return
    registry = ModelRegistry(settings.artifacts_dir)
    book = PositionBook(settings.state_dir / "positions.json")
    mgr = OrderManager(client, book)
    await asyncio.to_thread(mgr.reconcile, cfg.symbols)
    asyncio.create_task(run_user_stream(client, AccountState(), settings.binance_testnet))
    last_retrain = datetime.now(UTC)
    log.info("bot.started", testnet=settings.binance_testnet, symbols=cfg.symbols)

    while True:
        target = next_close(datetime.now(UTC))
        await asyncio.sleep(max((target + GRACE - datetime.now(UTC)).total_seconds(), 0))
        try:
            for s in cfg.symbols:  # make sure the just-closed candle and metrics are in the DB
                await asyncio.to_thread(backfill_symbol, engine, s, cfg.history_start)
            equity = await asyncio.to_thread(client.get_equity)
            reason = ks.check(equity)
            if reason:
                ks.lock(reason)  # lock first, then flatten
                log.critical("KILL_SWITCH_TRIGGERED", reason=reason, equity=equity)
                await asyncio.to_thread(mgr.close_all, cfg.symbols)
                log.critical("kill_switch.halted_manual_reset_required; also revoke/rotate API keys")
                return
            for s in cfg.symbols:
                await asyncio.to_thread(mgr.tick_bar, s, cfg.labeling.horizon_candles)
            for s in cfg.symbols:
                if s in book.trades:
                    continue
                await asyncio.to_thread(decide_and_trade, engine, cfg, registry, mgr, s,
                                        equity, len(book.trades), target)
            if datetime.now(UTC) - last_retrain >= timedelta(days=7):
                for s in cfg.symbols:
                    await asyncio.to_thread(retrain_symbol, engine, cfg, registry, s)
                last_retrain = datetime.now(UTC)
        except Exception:
            log.exception("cycle.failed")
