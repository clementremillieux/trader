"""Binance Handler for managing account and market data."""

import math

from typing import Any, Dict, List, Optional

import httpx

import pandas as pd

from binance.spot import Spot

from pydantic import BaseModel

from app.trader.schemas import PortfolioValue, Position

from config.logger_config import logger

API_KEY = "RRx439X2aBvHzrodPRhgNPAw9hyr48lYFqenjNIjWql25a9kuMMcdV7dRnjE9YsU"

API_SECRET = "RKSfisReVgtRa1bMqrzJjAaVhZ6OjAW9ATdLK3XC9gcBkYuCmOvC9ms77od4OoVp"

BASE_URL = "https://api.binance.com"


class SymbolInfos(BaseModel):
    """
    **Symbol Information**
    __________
    **Description:**

    This class represents the information of a symbol in the Binance API.

    __________
    **Parameters:**

    - **symbol (str)** - The symbol name.
    - **lot_size (float)** - The lot size for the symbol.
    """

    symbol: str

    lot_size: float

    min_qty: float

    step_size: float


class BinanceHandler:
    """
    **Binance Handler**
    __________
    **Description:**

    This class handles the connection to the Binance API and provides methods for managing account and market data.

    __________
    **Parameters:**

    - **api_key (str)** - The API key for authentication.
    - **api_secret (str)** - The API secret for authentication.
    """

    def __init__(self, main_currency: str):
        """
        **Initialize Binance Handler**
        __________
        **Description:**
        Initializes the BinanceHandler with the provided API key and secret.
        __________
        **Parameters:**
        - **api_key (str)** - The API key for authentication.
        - **api_secret (str)** - The API secret for authentication.
        """

        self.main_currency: str = main_currency

        self.client = Spot(
            api_key=API_KEY,
            api_secret=API_SECRET,
            base_url="https://api.binance.com",
        )

        self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=30.0, http2=True)

    def get_symbols_dict(self) -> Dict[str, str]:
        """
        **Get Symbols Dictionary**
        __________
        **Description:**
        Retrieves a dictionary of symbols and their corresponding base assets from the Binance API.
        __________
        **Returns:**
        - **Dict[str, str]** - A dictionary where the keys are symbols and the values are base assets.
        """

        exchange_info: Dict[str, Any] = self.client.exchange_info()

        symbols = exchange_info.get("symbols", [])

        trading_pairs = [s for s in symbols if s.get("status") == "TRADING"]

        sym_base_asset: Dict[str, str] = {
            s["symbol"]: s["baseAsset"] for s in trading_pairs
        }

        return sym_base_asset

    def get_tickers_existing_both_main_currency_usdt(self) -> List[str]:
        """
        **Get Tickers**
        """

        exchange_info: Dict[str, Any] = self.client.exchange_info()

        symbols = exchange_info.get("symbols", [])

        trading_pairs = [s for s in symbols if s.get("status") == "TRADING"]

        tickers: List[str] = [s["symbol"] for s in trading_pairs]

        tickers_in_usdt = [ticker for ticker in tickers if ticker.endswith("USDT")]

        tickers_in_usdc = [ticker for ticker in tickers if ticker.endswith("USDC")]

        tickers_in_usdt_and_usdc = [
            ticker
            for ticker in tickers_in_usdc
            if ticker.replace("USDC", "USDT") in tickers_in_usdt
        ]

        return tickers_in_usdt_and_usdc

    def get_portfolio(self) -> PortfolioValue:
        """
        **Get Account Information**
        __________
        **Description:**
        Retrieves account information from the Binance API.
        __________
        **Returns:**
        - **PortfolioValue** - The account information.
        """

        user_asset: List[Dict[str, Any]] = self.client.user_asset()

        buying_power: float = next(
            (
                float(asset.get("free", 0.0))
                for asset in user_asset
                if asset.get("asset") == self.main_currency
            ),
            0.0,
        )

        total_value: float = buying_power

        for asset in user_asset:
            if asset.get("asset") == self.main_currency:
                continue

            ticker_price: Optional[float] = self.get_ticker_price(
                symbol=asset.get("asset", "")
            )

            if ticker_price is None:
                logger.error(
                    "BINANCE => Error retrieving ticker price for symbol '%s'",
                    asset.get("asset", ""),
                )
                continue

            total_value += float(asset.get("free", 0.0)) * ticker_price

        return PortfolioValue(
            total_value=total_value,
            buying_power=buying_power,
        )

    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """
        **Get Ticker Price**
        __________
        **Description:**
        Retrieves the current price of a specific ticker from the Binance API.
        __________
        **Parameters:**
        - **symbol (str)** - The trading pair symbol (e.g., 'BTCUSDT').
        __________
        **Returns:**
        - **float** - The current price of the ticker.
        """

        try:
            if not symbol.endswith(self.main_currency):
                symbol = symbol + self.main_currency

            ticker_price: Dict[str, Any] = self.client.ticker_price(symbol=symbol)

            return float(ticker_price.get("price", 0.0))

        except Exception as e:
            logger.error(
                "BINANCE => Error retrieving ticker price for symbol '%s': %s",
                symbol,
                str(e),
            )

            return None

    def get_positions(self) -> List[Position]:
        """
        **Get Positions**
        __________
        **Description:**
        Retrieves the current positions from the Binance API.
        __________
        **Returns:**
        - **List[Dict[str, Any]]** - The current positions.
        """

        positions_dict: List[Dict[str, Any]] = self.client.user_asset()

        return [
            Position(
                symbol=position.get("asset", ""),
                qty=float(position.get("free", 0.0)),
                price=self.get_ticker_price(symbol=position.get("asset", "")),
            )
            for position in positions_dict
            if float(position.get("free", 0.0)) > 0.0
            and position.get("asset") != self.main_currency
        ]

    def adjust_quantity(self, qty, min_qty, step_size):
        """
        **Adjust Quantity**
        __________
        **Description:**
        Adjusts the quantity of an asset to be compliant with the exchange's minimum quantity and step size.
        __________
        **Parameters:**
        - **qty (float)** - The quantity to adjust.
        - **min_qty (float)** - The minimum quantity allowed by the exchange.
        - **step_size (float)** - The step size for the asset.
        __________
        **Returns:**
        - **float** - The adjusted quantity.
        """

        precision = str(step_size)[::-1].find(".")

        allowed_qty = math.floor(qty / step_size) * step_size

        adjusted = max(allowed_qty, min_qty)

        return round(adjusted, precision)

    def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
    ):
        """
        **Submit Order**
        __________
        **Description:**
        Submits an order to the Binance API.
        __________
        **Parameters:**
        - **symbol (str)** - The trading pair symbol (e.g., 'BTCUSDT').
        - **side (str)** - The order side ('BUY' or 'SELL').
        - **order_type (str)** - The type of order ('LIMIT', 'MARKET', etc.).
        - **quantity (float)** - The quantity of the asset to trade.
        - **price (float, optional)** - The price for limit orders.
        __________
        **Returns:**
        - **dict** - The response from the Binance API.
        """

        try:
            if not symbol.endswith(self.main_currency):
                symbol = symbol + self.main_currency

            info = self.get_symbol_lot_size(symbol=symbol)

            adj_qty = self.adjust_quantity(quantity, info.min_qty, info.step_size)

            if not symbol.endswith(self.main_currency):
                symbol = symbol + self.main_currency

            logger.info(
                "BINANCE => Submitting %s order: symbol=%s, side=%s, qty=%s, price=%s",
                order_type,
                symbol,
                side,
                adj_qty,
                price,
            )

            return self.client.new_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=adj_qty,
                price=price,
                recvWindow=6000,
            )
        except Exception as e:
            logger.error(
                "BINANCE => Error submitting order for symbol '%s': %s",
                symbol,
                str(e),
            )

            return None

    def get_all_tickers(self) -> List[str]:
        """
        **Get All Tickers**
        __________
        **Description:**
        Retrieves all trading pairs available on the Binance exchange.
        __________
        **Returns:**
        - **List[str]** - A list of trading pairs available on the exchange.
        __________
        **Raises:**
        - **Exception** if there is an error retrieving the data.
        __________
        **Notes:**
        - This method retrieves the exchange information and counts the number of trading pairs for each quote asset.
        - It logs the number of trading pairs for each quote asset.
        - It returns a list of tickers for the specified main currency.
        __________
        """

        exchange_info: Dict[str, Any] = self.client.exchange_info()

        symbols = exchange_info.get("symbols", [])

        trading_pairs = [s for s in symbols if s.get("status") == "TRADING"]

        tickers: List[str] = [
            s["symbol"] for s in trading_pairs if s["quoteAsset"] == self.main_currency
        ]

        logger.info(
            "BINANCE => %d tickers available for quoteAsset=%s",
            len(tickers),
            self.main_currency,
        )

        return tickers

    async def _query(self, path: str, params: dict | None = None) -> list[list]:
        """Requête GET générique sur l'API Binance (endpoint public)."""

        url = path if path.startswith("http") else f"{BASE_URL}{path}"

        resp = await self._client.get(url, params=params)

        resp.raise_for_status()

        return resp.json()

    async def klines(self, symbol: str, interval: str, **kwargs) -> list[list]:
        """
        Kline/Candlestick Data  —  GET /api/v3/klines
        https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data
        """
        if not symbol or not interval:
            raise ValueError("symbol et interval sont obligatoires")

        params = {"symbol": symbol, "interval": interval, **kwargs}

        return await self._query("/api/v3/klines", params)

    @staticmethod
    def _klines_to_df(raw: List[List]) -> pd.DataFrame:
        cols = [
            "OpenTime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "CloseTime",
            "QuoteAssetVolume",
            "NumTrades",
            "TakerBuyBaseVolume",
            "TakerBuyQuoteVolume",
            "Ignore",
        ]

        df = pd.DataFrame(raw, columns=cols)
        df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)

        return df.set_index("OpenTime")[
            [
                "Close",
                "Volume",
                "High",
                "Low",
                "Open",
                "QuoteAssetVolume",
                "NumTrades",
                "TakerBuyBaseVolume",
                "TakerBuyQuoteVolume",
            ]
        ].astype(float)

    async def get_historical_data_v2(
        self, ticker: str, interval: str, max_value: Optional[int] = None
    ) -> pd.DataFrame:
        """
        **Get Historical Data**
        """

        ticker_pair: str = ticker

        stock: List[List] = await self.klines(
            symbol=ticker_pair,
            interval=interval,
            limit=1000,
        )

        full_stock: List[List] = stock

        while True:
            stock = await self.klines(
                symbol=ticker_pair,
                interval=interval,
                limit=1000,
                startTime=stock[0][0] - 1000 * 60 * 60 * 24,
            )

            full_stock = stock + full_stock

            if len(stock) < 1000:
                break

            if max_value is not None and len(full_stock) >= max_value:
                full_stock = full_stock[-max_value:]
                break

        return self._klines_to_df(full_stock)

    def get_symbol_lot_size(self, symbol: str) -> SymbolInfos:
        """
        **Get Symbol Lot Size**
        __________
        **Description:**
        Retrieves the lot size for a specific symbol from the Binance API.
        __________
        **Parameters:**
        - **symbol (str)** - The trading pair symbol (e.g., 'BTCUSDT').
        __________
        **Returns:**
        - **SymbolInfos** - An object containing:
            - **symbol (str)**
            - **lot_size (float)**
            - **min_qty (float)**
            - **step_size (float)**
        __________
        **Raises:**
        - **ValueError** if the symbol isn’t returned by the API or if the LOT_SIZE filter is missing.
        """

        data = self.client.exchange_info(symbol)

        symbols = data.get("symbols", [])

        if not symbols:
            raise ValueError(f"No data returned for symbol '{symbol}'")

        symbol_info = symbols[0]

        filters = symbol_info.get("filters", [])

        lot_filter = next(
            (f for f in filters if f.get("filterType") == "LOT_SIZE"), None
        )

        if lot_filter is None:
            raise ValueError(f"LOT_SIZE filter not found for symbol '{symbol}'")

        min_qty = float(lot_filter["minQty"])

        step_size = float(lot_filter["stepSize"])

        return SymbolInfos(
            symbol=symbol,
            lot_size=min_qty,
            min_qty=min_qty,
            step_size=step_size,
        )
