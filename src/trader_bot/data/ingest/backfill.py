from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import structlog
from sqlalchemy.engine import Engine

from trader_bot.data.sources import api_bricks, binance_archive
from trader_bot.data.sources.binance_rest import INTERVAL_MS, BinanceMarketData, dt_to_ms
from trader_bot.db import schema, store

log = structlog.get_logger()
KLINE_INTERVALS = ("1h", "4h", "1d")


def find_gaps(index: pd.DatetimeIndex, step: pd.Timedelta) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (last_present, next_present) pairs where consecutive timestamps skip a step."""
    if len(index) < 2:
        return []
    idx = index.sort_values()
    diffs = idx[1:] - idx[:-1]
    return [(idx[i], idx[i + 1]) for i in range(len(diffs)) if diffs[i] > step]


def _start_for(engine: Engine, table, col, symbol, default: datetime, extra=None) -> datetime:
    last = store.last_ts(engine, table, col, symbol, extra)
    return last.to_pydatetime() if last is not None else default


def backfill_symbol(engine: Engine, symbol: str, history_start: str, md: BinanceMarketData | None = None) -> None:
    md = md or BinanceMarketData()
    default = datetime.fromisoformat(history_start).replace(tzinfo=UTC)
    end_ms = int(datetime.now(UTC).timestamp() * 1000)

    for interval in KLINE_INTERVALS:
        start = _start_for(engine, schema.klines, "open_time", symbol, default, {"interval": interval})
        n = store.upsert(
            engine, schema.klines, list(md.klines(symbol, interval, dt_to_ms(start), end_ms))
        )
        log.info("backfill.klines", symbol=symbol, interval=interval, rows=n)

    start = _start_for(engine, schema.funding, "ts", symbol, default)
    n = store.upsert(engine, schema.funding, list(md.funding_rates(symbol, dt_to_ms(start), end_ms)))
    log.info("backfill.funding", symbol=symbol, rows=n)

    start = _start_for(engine, schema.oi_metrics, "ts", symbol, default)
    total = 0
    log.info("backfill.metrics.started", symbol=symbol, start=start.date(), end=date.today())
    for rows in binance_archive.iter_metrics(symbol, start.date(), date.today()):
        total += store.upsert(engine, schema.oi_metrics, rows)
        log.info("backfill.metrics.batch_saved", symbol=symbol, rows=len(rows), total_rows=total)
    total += store.upsert(engine, schema.oi_metrics, md.recent_metrics(symbol))
    log.info("backfill.metrics", symbol=symbol, rows=total)


def backfill_liquidations(
    engine: Engine,
    symbols: list[str],
    start: date,
    end: date,
    api_key: str = "",
) -> int:
    """Load API BRICKS 4h liquidation aggregates into the existing table."""
    total = 0
    for symbol, rows in api_bricks.iter_liquidations(symbols, start, end, api_key):
        saved = store.upsert(engine, schema.liquidations, rows)
        total += saved
        log.info(
            "backfill.liquidations.api_bricks_symbol",
            symbol=symbol,
            rows=saved,
            total_rows=total,
        )
    log.info("backfill.liquidations.api_bricks_complete", rows=total)
    return total


def report_gaps(engine: Engine, symbol: str) -> dict[str, list]:
    out: dict[str, list] = {}
    for interval in KLINE_INTERVALS:
        df = store.load_frame(engine, schema.klines, "open_time", symbol, extra={"interval": interval})
        step = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
        out[f"klines_{interval}"] = find_gaps(df.index, step)
    m = store.load_frame(engine, schema.oi_metrics, "ts", symbol)
    out["oi_metrics"] = find_gaps(m.index, pd.Timedelta(minutes=5))
    return out
