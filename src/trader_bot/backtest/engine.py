"""Bar-based futures backtester using the same risk engine as live trading.

Timing: a decision made at the close of candle i-1 (index value t_i == open time of candle i)
is executed at the open of candle i. Stops are checked before targets within a bar
(conservative). Costs: taker fee on entry/exit, slippage against the trade, funding payments.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trader_bot.config.settings import BacktestCfg, RiskCfg
from trader_bot.risk.engine import Rejection, make_signal, plan_trade

BAR = pd.Timedelta(hours=4)


@dataclass
class _Pos:
    direction: int
    qty: float
    entry: float
    stop: float
    tp: float
    opened_at: pd.Timestamp
    bars: int = 0
    fees: float = 0.0
    funding: float = 0.0


def simulate(
    bars: pd.DataFrame,        # 4h candles indexed by open time: open/high/low/close
    preds: pd.DataFrame,       # indexed by decision time: p_short,p_flat,p_long,pred_ret
    atr: pd.Series,            # ATR known at decision time (same index as preds)
    funding: pd.Series,        # funding rate indexed by settlement time
    risk: RiskCfg,
    bt: BacktestCfg,
    confidence_threshold: float,
    horizon: int,
    initial_equity: float,
    diagnostics: dict | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    fee, slip = bt.taker_fee, bt.slippage_bps / 10_000
    cash = initial_equity
    pos: _Pos | None = None
    trades: list[dict] = []
    equity_points: dict[pd.Timestamp, float] = {}
    day_anchor: tuple | None = None
    week_anchor: tuple | None = None
    diag = diagnostics if diagnostics is not None else {}
    diag.update({
        "prediction_rows": int(len(preds)),
        "predictions_without_bar": int((~preds.index.isin(bars.index)).sum()),
        "position_open_skips": 0,
        "risk_halted": 0,
        "evaluated_predictions": 0,
        "top_class_counts": {"short": 0, "flat": 0, "long": 0},
        "directional_top_class": 0,
        "confidence_pass": 0,
        "return_agreement": 0,
        "signal_pass": 0,
        "risk_plan_approved": 0,
        "risk_plan_rejections": {},
    })

    bars = bars.loc[preds.index[0]:] if len(preds) else bars.iloc[0:0]

    def close_pos(price: float, t: pd.Timestamp, reason: str) -> None:
        nonlocal cash, pos
        p = pos
        exit_fee = p.qty * price * fee
        gross = p.direction * (price - p.entry) * p.qty
        pnl = gross - p.fees - exit_fee + p.funding
        cash += pnl
        trades.append({
            "opened_at": p.opened_at, "closed_at": t, "direction": p.direction, "qty": p.qty,
            "entry": p.entry, "exit": price, "pnl": pnl, "reason": reason,
        })
        pos = None

    for t, bar in bars.iterrows():
        # entry at the open of this bar from the decision made at the previous close
        if t in preds.index and pos is not None:
            diag["position_open_skips"] += 1
        if pos is None and t in preds.index and t in atr.index:
            diag["evaluated_predictions"] += 1
            d_key, w_key = t.date(), (t.isocalendar().year, t.isocalendar().week)
            if day_anchor is None or day_anchor[0] != d_key:
                day_anchor = (d_key, cash)
            if week_anchor is None or week_anchor[0] != w_key:
                week_anchor = (w_key, cash)
            halted = (
                (day_anchor[1] - cash) / day_anchor[1] >= risk.daily_loss_limit
                or (week_anchor[1] - cash) / week_anchor[1] >= risk.weekly_loss_limit
            )
            row = preds.loc[t]
            class_idx = [row["p_short"], row["p_flat"], row["p_long"]].index(
                max(row["p_short"], row["p_flat"], row["p_long"])
            )
            class_name = ("short", "flat", "long")[class_idx]
            diag["top_class_counts"][class_name] += 1
            direction = (-1, 0, 1)[class_idx]
            if direction != 0:
                diag["directional_top_class"] += 1
                if float(row[f"p_{class_name}"]) >= confidence_threshold:
                    diag["confidence_pass"] += 1
                    if float(row["pred_ret"]) * direction > 0:
                        diag["return_agreement"] += 1
            if halted:
                diag["risk_halted"] += 1
            sig = None if halted else make_signal(
                row["p_short"], row["p_flat"], row["p_long"], row["pred_ret"], confidence_threshold
            )
            if sig is not None:
                diag["signal_pass"] += 1
                entry = bar["open"] * (1 + sig.direction * slip)
                plan = plan_trade(sig, cash, entry, float(atr.loc[t]), risk)
                if isinstance(plan, Rejection):
                    reasons = diag["risk_plan_rejections"]
                    reasons[plan.reason] = reasons.get(plan.reason, 0) + 1
                else:
                    diag["risk_plan_approved"] += 1
                    pos = _Pos(plan.direction, plan.qty, entry, plan.stop, plan.take_profit, t)
                    pos.fees += plan.qty * entry * fee

        if pos is not None:
            pos.bars += 1
            f = funding.loc[(funding.index > t) & (funding.index <= t + BAR)]
            for rate in f:
                pos.funding -= pos.direction * rate * pos.qty * bar["close"]

            hit_stop = (bar["low"] <= pos.stop) if pos.direction == 1 else (bar["high"] >= pos.stop)
            hit_tp = (bar["high"] >= pos.tp) if pos.direction == 1 else (bar["low"] <= pos.tp)
            close_t = t + BAR
            if hit_stop:
                gap = min(bar["open"], pos.stop) if pos.direction == 1 else max(bar["open"], pos.stop)
                close_pos(gap * (1 - pos.direction * slip), close_t, "stop")
            elif hit_tp:
                close_pos(pos.tp, close_t, "take_profit")
            elif pos.bars >= horizon:
                close_pos(bar["close"] * (1 - pos.direction * slip), close_t, "time")

        unreal = pos.direction * (bar["close"] - pos.entry) * pos.qty if pos else 0.0
        equity_points[t + BAR] = cash + unreal

    return pd.Series(equity_points, dtype=float), pd.DataFrame(trades)
