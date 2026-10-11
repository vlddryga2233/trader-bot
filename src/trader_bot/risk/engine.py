"""Hard-coded risk rules. These run after the model and can only veto or shrink a trade."""
from __future__ import annotations

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
