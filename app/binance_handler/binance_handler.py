"""Binance Handler for managing account and market data."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from binance.spot import Spot

from app.trader.schemas import PortfolioValue, Position

from config.logger_config import logger

API_KEY = "RRx439X2aBvHzrodPRhgNPAw9hyr48lYFqenjNIjWql25a9kuMMcdV7dRnjE9YsU"

API_SECRET = "RKSfisReVgtRa1bMqrzJjAaVhZ6OjAW9ATdLK3XC9gcBkYuCmOvC9ms77od4OoVp"


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

    def __init__(self):
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

        self.client = Spot(
            api_key=API_KEY,
            api_secret=API_SECRET,
            base_url="https://api.binance.us",
        )

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
                if asset.get("asset") == "EUR"
            ),
            0.0,
        )

        total_value: float = sum(
            float(asset.get("free", 0.0))
            * self.get_ticker_price(symbol=asset.get("asset", ""))
            if asset.get("asset") != "EUR"
            else float(asset.get("free", 0.0))
            for asset in user_asset
        )

        return PortfolioValue(
            total_value=total_value,
            buying_power=buying_power,
        )

    def get_ticker_price(self, symbol: str) -> float:
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

        if not symbol.endswith("USDT"):
            symbol = symbol + "EUR"

        ticker_price: Dict[str, Any] = self.client.ticker_price(symbol=symbol)

        return float(ticker_price.get("price", 0.0))

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
            if float(position.get("free", 0.0)) > 0.0 and position.get("asset") != "EUR"
        ]

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

        return self.client.new_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            recvWindow=6000,
        )

    def get_all_tickers(self) -> List[str]:
        """
        **Get All Tickers**
        __________
        **Description:**
        Retrieves all available tickers from the Binance API.
        __________
        **Returns:**
        - **List[str]** - A list of all available tickers.
        """

        exchange_info: Dict[str, Any] = self.client.exchange_info()

        tickers: List[str] = [
            symbol["symbol"]
            for symbol in exchange_info.get("symbols", [])
            if symbol["status"] == "TRADING" and symbol["quoteAsset"] == "USDT"
        ]

        logger.info("BINANCE => %d tickers available", len(tickers))

        return tickers

    def get_historical_data(
        self,
        ticker: str,
        interval: str,
    ) -> pd.DataFrame:
        """
        Récupère l’historique sous forme de DataFrame, colonnes Close & Volume.
        Args:
            ticker (str): ex. 'BTCUSDT'
            interval (str): ex. '1m', '1h', '1d'
            days (int): nombre de jours à remonter
        Returns:
            pd.DataFrame | None: colonnes ['Close','Volume'], index UTC, ou None en cas d’erreur
        """

        if not ticker.endswith("USDT"):
            ticker += "USDT"

        raw = self.client.klines(
            symbol=ticker,
            interval=interval,
            limit=1000,
        )

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

        df = df.set_index("OpenTime")

        df = df[["Close", "Volume", "High", "Low"]].astype(float)

        return df
