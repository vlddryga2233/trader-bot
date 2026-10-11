"""Public Binance USD-M futures market-data REST client (no auth needed)."""
from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_PROD = "https://fapi.binance.com"
INTERVAL_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000, "5m": 300_000}


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class BinanceMarketData:
    def __init__(self, base_url: str = BASE_PROD, client: httpx.Client | None = None):
        self._http = client or httpx.Client(base_url=base_url, timeout=20)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30), reraise=True)
    def _get(self, path: str, params: dict[str, Any]) -> Any:
        r = self._http.get(path, params=params)
        if r.status_code in (418, 429):
            time.sleep(int(r.headers.get("Retry-After", 5)))
        r.raise_for_status()
        return r.json()

    def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> Iterator[dict]:
        """Yield kline rows in [start_ms, end_ms). Only fully closed candles are returned."""
        now_ms = int(time.time() * 1000)
        cursor = start_ms
        step = INTERVAL_MS[interval]
        while cursor < end_ms:
            batch = self._get(
                "/fapi/v1/klines",
                {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": 1500},
            )
            if not batch:
                break
            for k in batch:
                if k[0] + step > now_ms or k[0] >= end_ms:
                    return
                yield {
                    "symbol": symbol, "interval": interval, "open_time": ms_to_dt(k[0]),
                    "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                    "close": float(k[4]), "volume": float(k[5]), "quote_volume": float(k[7]),
                    "trades": int(k[8]), "taker_buy_base": float(k[9]),
                    "taker_buy_quote": float(k[10]),
                }
            cursor = batch[-1][0] + step

    def funding_rates(self, symbol: str, start_ms: int, end_ms: int) -> Iterator[dict]:
        cursor = start_ms
        while cursor < end_ms:
            batch = self._get(
                "/fapi/v1/fundingRate",
                {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            )
            if not batch:
                break
            for f in batch:
                yield {
                    "symbol": symbol, "ts": ms_to_dt(f["fundingTime"]),
                    "rate": float(f["fundingRate"]),
                    "mark_price": float(f["markPrice"]) if f.get("markPrice") else None,
                }
            cursor = batch[-1]["fundingTime"] + 1

    def recent_metrics(self, symbol: str, period: str = "5m", limit: int = 30) -> list[dict]:
        """Latest derivatives metrics (Binance keeps only ~30 days of this history)."""
        p = {"symbol": symbol, "period": period, "limit": limit}
        oi = self._get("/futures/data/openInterestHist", p)
        top_acc = self._get("/futures/data/topLongShortAccountRatio", p)
        top_pos = self._get("/futures/data/topLongShortPositionRatio", p)
        glob = self._get("/futures/data/globalLongShortAccountRatio", p)
        taker = self._get("/futures/data/takerlongshortRatio", p)

        rows: dict[int, dict] = {}

        def put(items: list[dict], mapping: dict[str, str]) -> None:
            for it in items:
                row = rows.setdefault(it["timestamp"], {"symbol": symbol})
                for src, dst in mapping.items():
                    row[dst] = float(it[src])

        put(oi, {"sumOpenInterest": "oi", "sumOpenInterestValue": "oi_value"})
        put(top_acc, {"longShortRatio": "top_ls_accounts"})
        put(top_pos, {"longShortRatio": "top_ls_positions"})
        put(glob, {"longShortRatio": "global_ls"})
        put(taker, {"buySellRatio": "taker_ls_vol"})
        metric_columns = (
            "oi", "oi_value", "top_ls_accounts", "top_ls_positions", "global_ls",
            "taker_ls_vol",
        )
        return [
            {
                "symbol": symbol,
                "ts": ms_to_dt(ts),
                **{column: row.get(column) for column in metric_columns},
            }
            for ts, row in sorted(rows.items())
        ]
