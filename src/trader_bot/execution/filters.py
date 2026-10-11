from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal


@dataclass(frozen=True)
class SymbolFilters:
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float


def parse_filters(symbol_info: dict) -> SymbolFilters:
    by_type = {f["filterType"]: f for f in symbol_info["filters"]}
    lot = by_type.get("MARKET_LOT_SIZE") or by_type["LOT_SIZE"]
    notional = by_type.get("MIN_NOTIONAL", {})
    return SymbolFilters(
        tick_size=float(by_type["PRICE_FILTER"]["tickSize"]),
        step_size=float(lot["stepSize"]),
        min_qty=float(lot["minQty"]),
        min_notional=float(notional.get("notional", notional.get("minNotional", 0)) or 0),
    )


def round_qty(qty: float, step: float) -> float:
    """Round down to the lot step (never exceed the risk-sized quantity)."""
    d = Decimal(str(step))
    return float((Decimal(str(qty)) / d).to_integral_value(rounding=ROUND_DOWN) * d)


def round_price(price: float, tick: float) -> float:
    d = Decimal(str(tick))
    return float((Decimal(str(price)) / d).to_integral_value(rounding=ROUND_HALF_UP) * d)


def valid_order(qty: float, price: float, f: SymbolFilters) -> str | None:
    if qty <= 0 or qty < f.min_qty or math.isclose(qty, 0):
        return f"qty {qty} below min_qty {f.min_qty}"
    if qty * price < f.min_notional:
        return f"notional {qty * price:.2f} below min_notional {f.min_notional}"
    return None
