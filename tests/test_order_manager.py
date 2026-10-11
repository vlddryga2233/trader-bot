import pytest

from trader_bot.execution.filters import SymbolFilters
from trader_bot.execution.manager import OrderManager, PositionBook
from trader_bot.risk.engine import TradePlan


class FakeClient:
    def __init__(self, fail_protective: bool = False):
        self.position = 0.0
        self.calls: list[tuple] = []
        self.fail_protective = fail_protective
        self.algo = 0

    def symbol_filters(self, symbol):
        return SymbolFilters(0.1, 0.001, 0.001, 1.0)

    def prepare_symbol(self, symbol, leverage):
        self.calls.append(("prepare", leverage))

    def market_order(self, symbol, side, qty, client_id, reduce_only=False):
        self.calls.append(("market", side, qty, reduce_only))
        self.position += qty if side == "BUY" else -qty
        return 100.0

    def protective_order(self, symbol, side, kind, trigger, client_id):
        if self.fail_protective:
            raise RuntimeError("boom")
        self.algo += 1
        self.calls.append((kind, side, trigger))

    def cancel_all(self, symbol):
        self.algo = 0
        self.calls.append(("cancel_all",))

    def get_position(self, symbol):
        return self.position

    def protective_order_count(self, symbol):
        return self.algo


PLAN = TradePlan(direction=1, qty=0.0159, entry=100.0, stop=98.04, take_profit=105.06,
                 leverage=3, risk_amount=30.0)


def test_open_trade_places_entry_stop_and_tp(tmp_path):
    c = FakeClient()
    m = OrderManager(c, PositionBook(tmp_path / "p.json"))
    assert m.open_trade("BTCUSDT", PLAN)
    assert ("market", "BUY", 0.015, False) in c.calls            # qty rounded down to step
    assert ("STOP_MARKET", "SELL", 98.0) in c.calls
    assert ("TAKE_PROFIT_MARKET", "SELL", 105.1) in c.calls
    assert "BTCUSDT" in PositionBook(tmp_path / "p.json").trades   # persisted


def test_unprotected_position_is_closed_immediately(tmp_path):
    c = FakeClient(fail_protective=True)
    m = OrderManager(c, PositionBook(tmp_path / "p.json"))
    assert m.open_trade("BTCUSDT", PLAN) is False
    assert c.position == 0
    assert "BTCUSDT" not in m.book.trades


def test_time_exit_after_horizon(tmp_path):
    c = FakeClient()
    m = OrderManager(c, PositionBook(tmp_path / "p.json"))
    m.open_trade("BTCUSDT", PLAN)
    for _ in range(2):
        m.tick_bar("BTCUSDT", horizon=3)
    assert c.position != 0
    m.tick_bar("BTCUSDT", horizon=3)
    assert c.position == 0 and "BTCUSDT" not in m.book.trades


def test_exchange_side_exit_clears_book(tmp_path):
    c = FakeClient()
    m = OrderManager(c, PositionBook(tmp_path / "p.json"))
    m.open_trade("BTCUSDT", PLAN)
    c.position = 0  # stop fired on the exchange
    m.tick_bar("BTCUSDT", horizon=6)
    assert "BTCUSDT" not in m.book.trades and ("cancel_all",) in c.calls


def test_reconcile_flattens_unprotected_position(tmp_path):
    c = FakeClient()
    c.position = 0.5
    m = OrderManager(c, PositionBook(tmp_path / "p.json"))
    m.reconcile(["BTCUSDT"])
    assert c.position == 0


def test_too_small_order_rejected(tmp_path):
    c = FakeClient()
    m = OrderManager(c, PositionBook(tmp_path / "p.json"))
    tiny = TradePlan(1, 0.0004, 100.0, 98.0, 105.0, 3, 1.0)
    assert m.open_trade("BTCUSDT", tiny) is False and c.position == 0
