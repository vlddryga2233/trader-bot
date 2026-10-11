"""User-data stream: live balance and order updates over WebSocket."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import structlog
import websockets

from trader_bot.execution.client import ExchangeClient

log = structlog.get_logger()
WS_PROD = "wss://fstream.binance.com/ws/"
WS_TESTNET = "wss://fstream.binancefuture.com/ws/"


@dataclass
class AccountState:
    wallet_balance: float | None = None
    order_status: dict[str, str] = field(default_factory=dict)


def apply_event(state: AccountState, ev: dict) -> None:
    kind = ev.get("e")
    if kind == "ACCOUNT_UPDATE":
        for b in ev["a"].get("B", []):
            if b["a"] == "USDT":
                state.wallet_balance = float(b["wb"])
    elif kind == "ORDER_TRADE_UPDATE":
        o = ev["o"]
        state.order_status[o["c"]] = o["X"]
        log.info("order.update", symbol=o["s"], client_id=o["c"], status=o["X"], side=o["S"])
    elif kind == "ALGO_UPDATE":
        log.info("algo.update", detail=ev.get("o"))


async def _keepalive(client: ExchangeClient, every_s: int = 1800) -> None:
    while True:
        await asyncio.sleep(every_s)
        await asyncio.to_thread(client.keepalive_user_stream)


async def run_user_stream(client: ExchangeClient, state: AccountState, testnet: bool) -> None:
    base = WS_TESTNET if testnet else WS_PROD
    backoff = 1
    while True:
        key = await asyncio.to_thread(client.start_user_stream)
        keep = asyncio.create_task(_keepalive(client))
        try:
            async with websockets.connect(base + key, ping_interval=20) as ws:
                backoff = 1
                async for raw in ws:
                    ev = json.loads(raw)
                    if ev.get("e") == "listenKeyExpired":
                        break
                    apply_event(state, ev)
        except Exception as e:
            log.warning("user_stream.reconnect", error=str(e), backoff=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
        finally:
            keep.cancel()
