"""Pluggable external-data adapters. Stubs are enabled only when an API key is configured."""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class OnchainProvider(Protocol):
    name: str

    def fetch(self, metric: str, asset: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
        """Return rows: {source, metric, asset, ts, value}."""


class LiquidationProvider(Protocol):
    name: str

    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
        """Return rows: {symbol, ts, side, price, qty, usd_value}."""


ONCHAIN_METRICS = [
    "exchange_inflow", "exchange_outflow", "whale_tx_count", "miner_outflow", "nupl", "mvrv_z",
]


class _NotImplementedProvider:
    name = "stub"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.name} adapter is a stub: implement fetch() against the provider API"
        )


class GlassnodeProvider(_NotImplementedProvider):
    name = "glassnode"


class CryptoQuantProvider(_NotImplementedProvider):
    name = "cryptoquant"


class CoinglassProvider(_NotImplementedProvider):
    name = "coinglass"


def build_onchain_provider(glassnode_key: str, cryptoquant_key: str) -> OnchainProvider | None:
    if glassnode_key:
        return GlassnodeProvider(glassnode_key)
    if cryptoquant_key:
        return CryptoQuantProvider(cryptoquant_key)
    return None


def build_liquidation_provider(coinglass_key: str) -> LiquidationProvider | None:
    return CoinglassProvider(coinglass_key) if coinglass_key else None
