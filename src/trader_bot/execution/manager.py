from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog

from trader_bot.execution.client import ExchangeClient
from trader_bot.execution.filters import round_price, round_qty, valid_order
from trader_bot.risk.engine import TradePlan

log = structlog.get_logger()


@dataclass
class OpenTrade:
    symbol: str
    direction: int
    qty: float
    entry: float
    stop: float
    take_profit: float
    opened_ts: float
    bars_held: int = 0


class PositionBook:
    """Persists the bot's own open trades so a restart can resume time-based exits."""

    def __init__(self, path: Path):
        self.path = path
        self.trades: dict[str, OpenTrade] = {}
        if path.exists():
            self.trades = {k: OpenTrade(**v) for k, v in json.loads(path.read_text()).items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({k: asdict(v) for k, v in self.trades.items()}))


class OrderManager:
    def __init__(self, client: ExchangeClient, book: PositionBook):
        self.client = client
        self.book = book

    @staticmethod
    def _cid(symbol: str, tag: str) -> str:
        return f"tb{int(time.time())}{symbol[:3]}{tag}"[:36]

    def open_trade(self, symbol: str, plan: TradePlan) -> bool:
        """Entry + protective stop and take-profit. An unprotected position is never left open."""
        f = self.client.symbol_filters(symbol)
        qty = round_qty(plan.qty, f.step_size)
        problem = valid_order(qty, plan.entry, f)
        if problem:
            log.warning("order.rejected", symbol=symbol, reason=problem)
            return False
        stop = round_price(plan.stop, f.tick_size)
        tp = round_price(plan.take_profit, f.tick_size)
        entry_side, exit_side = ("BUY", "SELL") if plan.direction == 1 else ("SELL", "BUY")

        self.client.prepare_symbol(symbol, plan.leverage)
        fill = self.client.market_order(symbol, entry_side, qty, self._cid(symbol, "E"))
        try:
            self.client.protective_order(symbol, exit_side, "STOP_MARKET", stop, self._cid(symbol, "S"))
            self.client.protective_order(symbol, exit_side, "TAKE_PROFIT_MARKET", tp, self._cid(symbol, "T"))
        except Exception:
            log.critical("protective_orders_failed_closing_position", symbol=symbol)
            self.emergency_close(symbol)
            return False

        self.book.trades[symbol] = OpenTrade(symbol, plan.direction, qty, fill or plan.entry,
                                             stop, tp, time.time())
        self.book.save()
        log.info("trade.opened", symbol=symbol, direction=plan.direction, qty=qty, stop=stop, tp=tp)
        return True

    def emergency_close(self, symbol: str) -> None:
        self.client.cancel_all(symbol)
        pos = self.client.get_position(symbol)
        if pos != 0:
            side = "SELL" if pos > 0 else "BUY"
            self.client.market_order(symbol, side, abs(pos), self._cid(symbol, "X"), reduce_only=True)
        self.book.trades.pop(symbol, None)
        self.book.save()
        log.warning("position.closed", symbol=symbol)

    def close_all(self, symbols: list[str]) -> None:
        for s in symbols:
            try:
                self.emergency_close(s)
            except Exception:
                log.exception("close_all.failed", symbol=s)

    def reconcile(self, symbols: list[str]) -> None:
        """Startup safety: drop stale book entries; flatten any position lacking protective orders."""
        for s in symbols:
            pos = self.client.get_position(s)
            if pos == 0:
                if self.book.trades.pop(s, None):
                    log.info("reconcile.stale_book_entry_removed", symbol=s)
                continue
            if self.client.protective_order_count(s) == 0 or s not in self.book.trades:
                log.critical("reconcile.unprotected_position_closing", symbol=s, qty=pos)
                self.emergency_close(s)
        self.book.save()

    def tick_bar(self, symbol: str, horizon: int) -> None:
        """Called once per closed candle: handle exchange-side exits and the time-based exit."""
        t = self.book.trades.get(symbol)
        if t is None:
            return
        if self.client.get_position(symbol) == 0:  # stop or take-profit already fired
            self.client.cancel_all(symbol)
            self.book.trades.pop(symbol, None)
            self.book.save()
            log.info("trade.closed_by_exchange", symbol=symbol)
            return
        t.bars_held += 1
        if t.bars_held >= horizon:
            log.info("trade.time_exit", symbol=symbol)
            self.emergency_close(symbol)
        else:
            self.book.save()
