"""Hard-coded risk rules. These run after the model and can only veto or shrink a trade."""
from __future__ import annotations

import math
from dataclasses import dataclass

from trader_bot.config.settings import RiskCfg
from trader_bot.risk import limits


@dataclass(frozen=True)
class Signal:
    direction: int       # +1 long, -1 short
    confidence: float
    pred_ret: float      # model's expected fractional move over the horizon (signed)


@dataclass(frozen=True)
class TradePlan:
    direction: int
    qty: float
    entry: float
    stop: float
    take_profit: float
    leverage: int
    risk_amount: float

    @property
    def notional(self) -> float:
        return self.qty * self.entry


@dataclass(frozen=True)
class Rejection:
    reason: str


def round_trip_cost(taker_fee: float, slippage_bps: float) -> float:
    """Entry + exit cost as a fraction of price."""
    return 2 * (taker_fee + slippage_bps / 10_000)


def make_ev_signal(p_long_win: float, p_short_win: float, price: float, atr: float,
                   stop_mult: float, tp_mult: float, cost_frac: float,
                   min_ev_r: float) -> Signal | None:
    """Choose a side only when its estimated net expected value clears the R threshold."""
    values = (p_long_win, p_short_win, price, atr, stop_mult, tp_mult, cost_frac, min_ev_r)
    if not all(math.isfinite(v) for v in values) or price <= 0 or atr <= 0:
        return None
    stop_dist = max(stop_mult, limits.MIN_ATR_STOP_MULT) * atr
    rr = tp_mult * atr / stop_dist
    cost_r = cost_frac * price / stop_dist
    ev_long = p_long_win * rr - (1 - p_long_win) - cost_r
    ev_short = p_short_win * rr - (1 - p_short_win) - cost_r
    if ev_long >= ev_short:
        direction, prob, ev = 1, p_long_win, ev_long
    else:
        direction, prob, ev = -1, p_short_win, ev_short
    if ev < min_ev_r:
        return None
    return Signal(direction, prob, direction * tp_mult * atr / price * (1 + 1e-9))


def make_signal(p_short: float, p_flat: float, p_long: float, pred_ret: float,
                threshold: float) -> Signal | None:
    """Directional signal only when the top class is long/short with confidence >= threshold
    and the regressor agrees on the sign."""
    probs = {-1: p_short, 0: p_flat, 1: p_long}
    direction = max(probs, key=probs.get)
    conf = probs[direction]
    if direction == 0 or conf < threshold:
        return None
    if pred_ret * direction <= 0:
        return None
    return Signal(direction, conf, pred_ret)


def plan_trade(signal: Signal, equity: float, price: float, atr: float, cfg: RiskCfg,
               open_positions: int = 0) -> TradePlan | Rejection:
    risk_pct = min(cfg.risk_per_trade, limits.MAX_RISK_PER_TRADE)
    leverage = min(cfg.leverage, limits.MAX_LEVERAGE)
    stop_mult = max(cfg.atr_stop_mult, limits.MIN_ATR_STOP_MULT)
    max_pos = min(cfg.max_open_positions, limits.MAX_OPEN_POSITIONS)

    if equity <= 0 or price <= 0 or atr is None or atr <= 0:
        return Rejection("invalid equity/price/atr")
    if open_positions >= max_pos:
        return Rejection("max open positions reached")

    stop_dist = stop_mult * atr
    if stop_dist >= price:
        return Rejection("stop distance >= price")
    tp_dist = abs(signal.pred_ret) * price
    if tp_dist < cfg.min_reward_risk * stop_dist:
        return Rejection("reward/risk below minimum")

    risk_amount = equity * risk_pct
    qty = risk_amount / stop_dist
    max_qty = equity * leverage / price  # leverage cap shrinks the size
    qty = min(qty, max_qty)
    risk_amount = qty * stop_dist
    d = signal.direction
    return TradePlan(
        direction=d, qty=qty, entry=price,
        stop=price - d * stop_dist, take_profit=price + d * tp_dist,
        leverage=leverage, risk_amount=risk_amount,
    )
