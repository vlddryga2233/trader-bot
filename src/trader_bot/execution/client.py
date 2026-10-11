from __future__ import annotations

import os
from typing import Any, Protocol
from urllib.parse import urlparse

import structlog

from trader_bot.execution.filters import SymbolFilters, parse_filters

log = structlog.get_logger()


class ExchangeClient(Protocol):
    def get_equity(self) -> float: ...
    def get_position(self, symbol: str) -> float: ...           # signed quantity
    def symbol_filters(self, symbol: str) -> SymbolFilters: ...
    def prepare_symbol(self, symbol: str, leverage: int) -> None: ...  # isolated margin + leverage
    def market_order(self, symbol: str, side: str, qty: float, client_id: str,
                     reduce_only: bool = False) -> float: ...   # returns average fill price
    def protective_order(self, symbol: str, side: str, kind: str, trigger: float,
                         client_id: str) -> None: ...            # close-position STOP/TP market
    def cancel_all(self, symbol: str) -> None: ...
    def protective_order_count(self, symbol: str) -> int: ...
    def start_user_stream(self) -> str: ...
    def keepalive_user_stream(self) -> None: ...


def _dump(o: Any) -> Any:
    if hasattr(o, "actual_instance") and o.actual_instance is not None:
        return _dump(o.actual_instance)
    if isinstance(o, list):
        return [_dump(x) for x in o]
    if hasattr(o, "model_dump"):
        return o.model_dump(by_alias=True)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return o


def _data(resp: Any) -> Any:
    return _dump(resp.data())


def _g(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


class BinanceSdkClient:
    """Adapter over binance-sdk-derivatives-trading-usds-futures (REST).

    Conditional orders (stop / take-profit) go through the Algo Order endpoints, which is where
    Binance moved them. Response field names are read defensively (alias or snake_case).
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        from binance_common.configuration import ConfigurationRestAPI
        from binance_common.constants import (
            DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
            DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
        )
        from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
            DerivativesTradingUsdsFutures,
        )

        base = (DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL if testnet
                else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL)
        kwargs: dict[str, Any] = {"api_key": api_key, "api_secret": api_secret,
                                  "base_path": base, "timeout": 10_000}
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            u = urlparse(proxy)
            kwargs["proxy"] = {"host": u.hostname, "port": u.port, "protocol": u.scheme}
        self.testnet = testnet
        self._api = DerivativesTradingUsdsFutures(config_rest_api=ConfigurationRestAPI(**kwargs)).rest_api
        self._filters: dict[str, SymbolFilters] = {}

    def get_equity(self) -> float:
        acc = _data(self._api.account_information_v3())
        return float(_g(acc, "totalMarginBalance", "total_margin_balance"))

    def get_position(self, symbol: str) -> float:
        rows = _data(self._api.position_information_v3(symbol=symbol))
        rows = rows if isinstance(rows, list) else [rows]
        return sum(float(_g(r, "positionAmt", "position_amt", default=0)) for r in rows)

    def symbol_filters(self, symbol: str) -> SymbolFilters:
        if symbol not in self._filters:
            info = _data(self._api.exchange_information())
            for s in info["symbols"]:
                self._filters[s["symbol"]] = parse_filters(s)
        return self._filters[symbol]

    def prepare_symbol(self, symbol: str, leverage: int) -> None:
        from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
            ChangeMarginTypeMarginTypeEnum,
        )

        try:
            self._api.change_margin_type(symbol=symbol, margin_type=ChangeMarginTypeMarginTypeEnum.ISOLATED)
        except Exception as e:  # "No need to change margin type" is not an error for us
            log.info("margin_type.unchanged", symbol=symbol, detail=str(e))
        self._api.change_initial_leverage(symbol=symbol, leverage=leverage)

    def market_order(self, symbol: str, side: str, qty: float, client_id: str,
                     reduce_only: bool = False) -> float:
        from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
            NewOrderNewOrderRespTypeEnum,
            NewOrderReduceOnlyEnum,
            NewOrderSideEnum,
            NewOrderTypeEnum,
        )

        kw: dict[str, Any] = {}
        if reduce_only:
            kw["reduce_only"] = NewOrderReduceOnlyEnum.TRUE
        r = _data(self._api.new_order(
            symbol=symbol, side=NewOrderSideEnum[side], type=NewOrderTypeEnum.MARKET,
            quantity=qty, new_client_order_id=client_id,
            new_order_resp_type=NewOrderNewOrderRespTypeEnum.RESULT, **kw,
        ))
        return float(_g(r, "avgPrice", "avg_price", default=0) or 0)

    def protective_order(self, symbol: str, side: str, kind: str, trigger: float,
                         client_id: str) -> None:
        from binance_sdk_derivatives_trading_usds_futures.rest_api.models import (
            NewAlgoOrderAlgoTypeEnum,
            NewAlgoOrderClosePositionEnum,
            NewAlgoOrderSideEnum,
            NewAlgoOrderTypeEnum,
            NewAlgoOrderWorkingTypeEnum,
        )

        self._api.new_algo_order(
            algo_type=NewAlgoOrderAlgoTypeEnum.CONDITIONAL, symbol=symbol,
            side=NewAlgoOrderSideEnum[side], type=NewAlgoOrderTypeEnum[kind],
            trigger_price=trigger, close_position=NewAlgoOrderClosePositionEnum.TRUE,
            working_type=NewAlgoOrderWorkingTypeEnum.MARK_PRICE, client_algo_id=client_id,
        )

    def cancel_all(self, symbol: str) -> None:
        self._api.cancel_all_open_orders(symbol=symbol)
        self._api.cancel_all_algo_open_orders(symbol=symbol)

    def protective_order_count(self, symbol: str) -> int:
        rows = _data(self._api.current_all_algo_open_orders(symbol=symbol))
        return len(rows if isinstance(rows, list) else [])

    def start_user_stream(self) -> str:
        return _g(_data(self._api.start_user_data_stream()), "listenKey", "listen_key")

    def keepalive_user_stream(self) -> None:
        self._api.keepalive_user_data_stream()
