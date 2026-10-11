"""Live collectors: closed klines + liquidations over WebSocket, derivatives metrics by polling."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import structlog
import websockets
from sqlalchemy.engine import Engine

from trader_bot.data.ingest.backfill import KLINE_INTERVALS
from trader_bot.data.sources.binance_rest import BinanceMarketData, ms_to_dt
from trader_bot.db import schema, store

log = structlog.get_logger()
WS_BASE = "wss://fstream.binance.com"


def parse_kline_event(msg: dict) -> dict | None:
    k = msg.get("k")
    if not k or not k.get("x"):  # only closed candles
        return None
    return {
        "symbol": k["s"], "interval": k["i"], "open_time": ms_to_dt(k["t"]),
        "open": float(k["o"]), "high": float(k["h"]), "low": float(k["l"]), "close": float(k["c"]),
        "volume": float(k["v"]), "quote_volume": float(k["q"]), "trades": int(k["n"]),
        "taker_buy_base": float(k["V"]), "taker_buy_quote": float(k["Q"]),
    }


def parse_liquidation_event(msg: dict) -> dict | None:
    o = msg.get("o")
    if not o:
        return None
    qty = float(o.get("z") or o["q"])
    price = float(o.get("ap") or o["p"])
    return {
        "symbol": o["s"], "ts": ms_to_dt(o["T"]),
        # a SELL liquidation order closes a LONG position and vice versa
        "side": "LONG" if o["S"] == "SELL" else "SHORT",
        "price": price, "qty": qty, "usd_value": qty * price,
    }


async def _listen(url: str, handler) -> None:
    backoff = 1
    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                backoff = 1
                async for raw in ws:
                    handler(json.loads(raw))
        except Exception as e:  # reconnect on any transport error
            log.warning("ws.reconnect", url=url, error=str(e), backoff=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def _poll_metrics(engine: Engine, symbols: list[str], every_s: int = 300) -> None:
    md = BinanceMarketData()
    while True:
        for s in symbols:
            try:
                rows = await asyncio.to_thread(md.recent_metrics, s, "5m", 12)
                store.upsert(engine, schema.oi_metrics, rows)
                end = int(datetime.now(UTC).timestamp() * 1000)
                f = await asyncio.to_thread(
                    lambda s=s: list(md.funding_rates(s, end - 9 * 3_600_000, end))
                )
                store.upsert(engine, schema.funding, f)
            except Exception as e:
                log.warning("poll.failed", symbol=s, error=str(e))
        await asyncio.sleep(every_s)


async def run_collectors(engine: Engine, symbols: list[str]) -> None:
    streams = "/".join(f"{s.lower()}@kline_{i}" for s in symbols for i in KLINE_INTERVALS)

    def on_kline(msg: dict) -> None:
        row = parse_kline_event(msg.get("data", msg))
        if row:
            store.upsert(engine, schema.klines, [row])
            log.info("kline.closed", symbol=row["symbol"], interval=row["interval"])

    watch = {s.upper() for s in symbols}

    def on_liq(msg: dict) -> None:
        row = parse_liquidation_event(msg)
        if row and row["symbol"] in watch:
            store.upsert(engine, schema.liquidations, [row])

    await asyncio.gather(
        _listen(f"{WS_BASE}/stream?streams={streams}", on_kline),
        _listen(f"{WS_BASE}/ws/!forceOrder@arr", on_liq),
        _poll_metrics(engine, symbols),
    )
