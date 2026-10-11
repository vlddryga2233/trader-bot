import numpy as np
import pandas as pd
import pytest

from trader_bot.backtest.engine import simulate
from trader_bot.backtest.metrics import compute_metrics, passes_gate
from trader_bot.config.settings import BacktestCfg, RiskCfg
from trader_bot.execution.filters import SymbolFilters, parse_filters, round_price, round_qty, valid_order

RISK = RiskCfg(risk_per_trade=0.01, leverage=3, atr_stop_mult=2.0, min_reward_risk=1.0,
               daily_loss_limit=0.5, weekly_loss_limit=0.5)
BT = BacktestCfg(taker_fee=0.0, slippage_bps=0.0, initial_equity=10_000)


def _bars(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="4h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def _long_pred(bars, i=0, pred_ret=0.05):
    t = bars.index[i]
    preds = pd.DataFrame({"p_short": 0.05, "p_flat": 0.05, "p_long": 0.9, "pred_ret": pred_ret}, index=[t])
    return preds, pd.Series([1.0], index=[t])  # atr = 1 -> stop distance 2


def _run(bars, **kw):
    preds, atr = _long_pred(bars, **kw)
    funding = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    return simulate(bars, preds, atr, funding, RISK, BT, 0.75, horizon=6, initial_equity=10_000)


def test_take_profit_hit_pnl_matches_risk_math():
    bars = _bars([[100, 101, 99.5, 100], [100, 106, 99.5, 105], [105, 105, 104, 105]])
    equity, trades = _run(bars)
    assert len(trades) == 1 and trades.iloc[0]["reason"] == "take_profit"
    # qty = 100/2 = 50; tp = 105 -> +250
    assert trades.iloc[0]["pnl"] == pytest.approx(250.0)
    assert equity.iloc[-1] == pytest.approx(10_250.0)


def test_stop_wins_when_stop_and_target_in_same_bar():
    bars = _bars([[100, 106, 97, 100], [100, 100, 100, 100]])
    _, trades = _run(bars)
    assert trades.iloc[0]["reason"] == "stop"
    assert trades.iloc[0]["pnl"] == pytest.approx(-100.0)   # exactly 1% of equity


def test_gap_through_stop_exits_at_open():
    bars = _bars([[100, 100.5, 99.5, 100], [95, 96, 94, 95], [95, 95, 95, 95]])
    _, trades = _run(bars)
    assert trades.iloc[0]["exit"] == pytest.approx(95.0)    # worse than the 98 stop
    assert trades.iloc[0]["pnl"] == pytest.approx(-250.0)


def test_time_exit_and_fees_reduce_pnl():
    bars = _bars([[100, 100.5, 99.5, 100]] * 8)
    preds, atr = _long_pred(bars)
    funding = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    bt = BacktestCfg(taker_fee=0.0005, slippage_bps=0.0, initial_equity=10_000)
    _, trades = simulate(bars, preds, atr, funding, RISK, bt, 0.75, horizon=6, initial_equity=10_000)
    assert trades.iloc[0]["reason"] == "time"
    assert trades.iloc[0]["pnl"] == pytest.approx(-(50 * 100 * 0.0005) * 2)


def test_funding_paid_by_longs_when_rate_positive():
    bars = _bars([[100, 100.5, 99.5, 100]] * 3)
    preds, atr = _long_pred(bars)
    funding = pd.Series([0.001], index=pd.DatetimeIndex([bars.index[1] + pd.Timedelta(hours=1)]))
    _, trades = simulate(bars, preds, atr, funding, RISK, BT, 0.75, horizon=2, initial_equity=10_000)
    assert trades.iloc[0]["pnl"] == pytest.approx(-0.001 * 50 * 100)


def test_no_trade_below_confidence():
    bars = _bars([[100, 101, 99, 100]] * 3)
    t = bars.index[0]
    preds = pd.DataFrame({"p_short": 0.2, "p_flat": 0.2, "p_long": 0.6, "pred_ret": 0.05}, index=[t])
    funding = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    _, trades = simulate(bars, preds, pd.Series([1.0], index=[t]), funding, RISK, BT, 0.75, 6, 10_000)
    assert trades.empty


def test_metrics_and_gate():
    idx = pd.date_range("2024-01-01", periods=730, freq="12h", tz="UTC")
    eq = pd.Series(10_000 * np.exp(np.linspace(0, 0.5, 730) + np.random.default_rng(0).normal(0, 0.002, 730)),
                   index=idx)
    m = compute_metrics(eq, pd.DataFrame({"pnl": [1.0, -0.5, 2.0]}))
    assert m["total_return"] > 0.5 and m["n_trades"] == 3 and m["profit_factor"] == pytest.approx(6.0)
    assert passes_gate({"sharpe": 2.0, "max_drawdown": -0.1})
    assert not passes_gate({"sharpe": 2.0, "max_drawdown": -0.2})
    assert not passes_gate({"sharpe": 1.0, "max_drawdown": -0.05})


def test_exchange_filters():
    info = {"symbol": "BTCUSDT", "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
        {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
        {"filterType": "MIN_NOTIONAL", "notional": "100"}]}
    f = parse_filters(info)
    assert f == SymbolFilters(0.1, 0.001, 0.001, 100.0)
    assert round_qty(0.0129, f.step_size) == 0.012           # always rounds down
    assert round_price(65000.04, f.tick_size) == 65000.0
    assert valid_order(0.0005, 60000, f) is not None
    assert valid_order(0.002, 60000, f) is None
    assert valid_order(0.001, 50000, f) is not None          # notional 50 < 100
