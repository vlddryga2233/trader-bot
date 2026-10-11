"""Historical liquidation metrics from the CoinAPI / API BRICKS Metrics V1 API."""
from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import httpx
import structlog

BASE = "https://rest.coinapi.io/v1/metrics/symbol/history"
SYMBOL_IDS = {
    "BTCUSDT": "BINANCEFTS_PERP_BTC_USDT",
    "ETHUSDT": "BINANCEFTS_PERP_ETH_USDT",
}
PERIOD_ID = "4HRS"
log = structlog.get_logger()


def _get_history(
    client: httpx.Client,
    symbol_id: str,
    metric_id: str,
    start: datetime,
    end: datetime,
) -> list[dict]:
    params = {
        "metric_id": metric_id,
        "symbol_id": symbol_id,
        "time_start": start.isoformat(),
        "time_end": end.isoformat(),
        "period_id": PERIOD_ID,
        "limit": 100_000,
    }
    for attempt in range(5):
        response = client.get(BASE, params=params)
        if response.status_code == 429 and attempt < 4:
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 2**attempt
            time.sleep(min(delay, 60))
            continue
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected API BRICKS response for {metric_id}")
        return payload
    return []


def iter_liquidations(
    symbols: list[str],
    start: date,
    end: date,
    api_key: str,
) -> Iterator[tuple[str, list[dict]]]:
    """Yield 4h total-liquidation rows for each symbol; side is intentionally unknown.

    USD notional is estimated as summed quantity times the average liquidation price.
    The API aggregates those metrics separately, so the product is an approximation.
    """
    if start > end:
        raise ValueError("start date must not be after end date")
    if not api_key:
        raise ValueError("API_BRIKS_API_KEY is required for API BRICKS history")

    start_ts = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    # End is exclusive in the API; include the full requested final calendar day.
    end_ts = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    headers = {"X-CoinAPI-Key": api_key}
    with httpx.Client(timeout=90, headers=headers) as client:
        for symbol in symbols:
            try:
                symbol_id = SYMBOL_IDS[symbol.upper()]
            except KeyError as exc:
                raise ValueError(f"No API BRICKS symbol mapping for {symbol}") from exc
            quantity_rows = _get_history(
                client, symbol_id, "LIQUIDATION_QUANTITY", start_ts, end_ts
            )
            price_rows = _get_history(
                client, symbol_id, "LIQUIDATION_AVERAGE_PRICE", start_ts, end_ts
            )
            prices = {row.get("time_period_start"): row for row in price_rows}
            observed: dict[datetime, dict] = {}
            missing_prices = 0
            for row in quantity_rows:
                try:
                    qty = float(row["sum"])
                    period_start = row["time_period_start"]
                    period_end = datetime.fromisoformat(
                        row["time_period_end"].replace("Z", "+00:00")
                    ).astimezone(UTC)
                    price_row = prices.get(period_start)
                    count = float(price_row.get("count", 0)) if price_row else 0
                    avg_price = float(price_row["sum"]) / count if count > 0 else 0.0
                except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                    log.warning(
                        "backfill.liquidations.api_bricks_row_skipped",
                        symbol=symbol,
                        period_start=row.get("time_period_start"),
                        reason=str(exc),
                    )
                    continue
                if qty < 0 or avg_price <= 0:
                    if qty < 0:
                        continue
                    missing_prices += 1
                observed[period_end] = {
                    "symbol": symbol.upper(),
                    "ts": period_end,
                    "side": "UNKNOWN",
                    "price": avg_price,
                    "qty": qty,
                    # Preserve quantity when the API has no usable price; a zero
                    # notional would falsely claim there was no liquidation.
                    "usd_value": qty * avg_price if avg_price > 0 else (None if qty > 0 else 0.0),
                }
            # Explicit zero buckets make absence meaningful only inside observed
            # API coverage; outside this range the feature remains missing.
            out: list[dict] = []
            if observed:
                first, last = min(observed), max(observed)
                period_end = first
                while period_end <= last:
                    out.append(observed.get(period_end, {
                        "symbol": symbol.upper(),
                        "ts": period_end,
                        "side": "UNKNOWN",
                        "price": 0.0,
                        "qty": 0.0,
                        "usd_value": 0.0,
                    }))
                    period_end += timedelta(hours=4)
            if missing_prices:
                log.warning(
                    "backfill.liquidations.api_bricks_prices_missing",
                    symbol=symbol,
                    rows=missing_prices,
                )
            yield symbol.upper(), out
