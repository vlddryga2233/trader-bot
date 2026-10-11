"""Historical derivatives metrics from the public data.binance.vision archive."""
from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta

import httpx
import structlog

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
log = structlog.get_logger()
ARCHIVE_BATCH_DAYS = 7
ARCHIVE_WORKERS = 8


def _f(v: str) -> float | None:
    return float(v) if v not in ("", None) else None


def parse_metrics_csv(symbol: str, text: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        ts = datetime.strptime(r["create_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        rows.append({
            "symbol": symbol, "ts": ts,
            "oi": _f(r["sum_open_interest"]), "oi_value": _f(r["sum_open_interest_value"]),
            "top_ls_accounts": _f(r["count_toptrader_long_short_ratio"]),
            "top_ls_positions": _f(r["sum_toptrader_long_short_ratio"]),
            "global_ls": _f(r["count_long_short_ratio"]),
            "taker_ls_vol": _f(r["sum_taker_long_short_vol_ratio"]),
        })
    return rows


def fetch_day(client: httpx.Client, symbol: str, day: date) -> list[dict] | None:
    url = f"{BASE}/{symbol}/{symbol}-metrics-{day.isoformat()}.zip"
    r = client.get(url)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        return parse_metrics_csv(symbol, z.read(z.namelist()[0]).decode())


def iter_metrics(symbol: str, start: date, end: date) -> Iterator[list[dict]]:
    """Yield combined rows in weekly batches from available daily archives."""
    if start > end:
        return

    total_days = (end - start).days + 1
    with httpx.Client(timeout=60) as client:
        with ThreadPoolExecutor(max_workers=ARCHIVE_WORKERS) as pool:
            batch_start = start
            completed_days = 0
            while batch_start <= end:
                batch_end = min(batch_start + timedelta(days=ARCHIVE_BATCH_DAYS - 1), end)
                days = [
                    batch_start + timedelta(days=i)
                    for i in range((batch_end - batch_start).days + 1)
                ]
                futures = {day: pool.submit(fetch_day, client, symbol, day) for day in days}
                batch_rows: list[dict] = []
                for day in days:
                    rows = futures[day].result()
                    if rows:
                        batch_rows.extend(rows)

                completed_days += len(days)
                log.info(
                    "backfill.metrics.archive_progress",
                    symbol=symbol,
                    days_completed=completed_days,
                    days_total=total_days,
                    rows=len(batch_rows),
                )
                if batch_rows:
                    yield batch_rows
                batch_start = batch_end + timedelta(days=1)
