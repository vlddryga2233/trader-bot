from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy import Table, create_engine, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

CHUNK = 5000


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def upsert(engine: Engine, table: Table, rows: Sequence[dict[str, Any]]) -> int:
    """Idempotent insert; duplicate primary keys in a batch keep their last row."""
    if not rows:
        return 0
    pk = [c.name for c in table.primary_key.columns]
    # PostgreSQL cannot update the same target row twice within one INSERT ... ON CONFLICT
    # statement. Archive files and merged live data can contain duplicate timestamps, so
    # collapse those keys before constructing any statement. Assignment preserves the
    # first-seen ordering while replacing its values with the last-seen row.
    unique_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[column] for column in pk)
        unique_rows[key] = row
    rows_to_write = list(unique_rows.values())
    with engine.begin() as conn:
        for i in range(0, len(rows_to_write), CHUNK):
            chunk = rows_to_write[i : i + CHUNK]
            stmt = insert(table).values(chunk)
            update_cols = {c: stmt.excluded[c] for c in chunk[0] if c not in pk}
            if update_cols:
                stmt = stmt.on_conflict_do_update(index_elements=pk, set_=update_cols)
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=pk)
            conn.execute(stmt)
    return len(rows)


def load_frame(
    engine: Engine,
    table: Table,
    time_col: str,
    symbol: str | None = None,
    since: pd.Timestamp | None = None,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load rows ordered by time with a UTC DatetimeIndex."""
    q = select(table)
    if symbol is not None and "symbol" in table.c:
        q = q.where(table.c.symbol == symbol)
    for k, v in (extra or {}).items():
        q = q.where(table.c[k] == v)
    if since is not None:
        q = q.where(table.c[time_col] >= since.to_pydatetime())
    q = q.order_by(table.c[time_col])
    df = pd.read_sql(q, engine)
    if df.empty:
        return df.set_index(pd.DatetimeIndex([], tz="UTC", name=time_col))
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    return df.set_index(time_col)


def last_ts(
    engine: Engine, table: Table, time_col: str, symbol: str, extra: dict[str, Any] | None = None
) -> pd.Timestamp | None:
    from sqlalchemy import func

    q = select(func.max(table.c[time_col])).where(table.c.symbol == symbol)
    for k, v in (extra or {}).items():
        q = q.where(table.c[k] == v)
    with engine.connect() as conn:
        val = conn.execute(q).scalar()
    return pd.Timestamp(val).tz_convert("UTC") if val is not None else None
