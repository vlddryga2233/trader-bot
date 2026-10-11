"""Schema (SQLAlchemy Core) and Timescale hypertable setup."""
from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

metadata = MetaData()
TS = DateTime(timezone=True)

klines = Table(
    "klines", metadata,
    Column("symbol", String, primary_key=True),
    Column("interval", String, primary_key=True),
    Column("open_time", TS, primary_key=True),
    Column("open", Float), Column("high", Float), Column("low", Float), Column("close", Float),
    Column("volume", Float), Column("quote_volume", Float), Column("trades", Integer),
    Column("taker_buy_base", Float), Column("taker_buy_quote", Float),
)

funding = Table(
    "funding", metadata,
    Column("symbol", String, primary_key=True),
    Column("ts", TS, primary_key=True),
    Column("rate", Float), Column("mark_price", Float),
)

# 5-minute derivatives metrics: open interest, long/short ratios, taker volume ratio
oi_metrics = Table(
    "oi_metrics", metadata,
    Column("symbol", String, primary_key=True),
    Column("ts", TS, primary_key=True),
    Column("oi", Float), Column("oi_value", Float),
    Column("top_ls_accounts", Float), Column("top_ls_positions", Float),
    Column("global_ls", Float), Column("taker_ls_vol", Float),
)

liquidations = Table(
    "liquidations", metadata,
    Column("symbol", String, primary_key=True),
    Column("ts", TS, primary_key=True),
    Column("side", String, primary_key=True),  # LONG / SHORT for live events; UNKNOWN for aggregates
    Column("price", Float, primary_key=True),
    Column("qty", Float), Column("usd_value", Float),
)

onchain = Table(
    "onchain", metadata,
    Column("source", String, primary_key=True),
    Column("metric", String, primary_key=True),
    Column("asset", String, primary_key=True),
    Column("ts", TS, primary_key=True),
    Column("value", Float),
)


signals = Table(
    "signals", metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("ts", TS), Column("symbol", String), Column("model_version", String),
    Column("direction", Integer), Column("confidence", Float), Column("pred_return", Float),
    Column("accepted", Boolean), Column("reason", Text),
)

orders = Table(
    "orders", metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("ts", TS), Column("symbol", String), Column("client_id", String, unique=True),
    Column("kind", String), Column("side", String), Column("qty", Float), Column("price", Float),
    Column("status", String), Column("raw", JSONB),
)




HYPERTABLES = {
    "klines": "open_time",
    "funding": "ts",
    "oi_metrics": "ts",
    "liquidations": "ts",
    "onchain": "ts",
}


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
        for table, col in HYPERTABLES.items():
            conn.execute(
                text(
                    f"SELECT create_hypertable('{table}', '{col}', "
                    "if_not_exists => TRUE, migrate_data => TRUE)"
                )
            )
