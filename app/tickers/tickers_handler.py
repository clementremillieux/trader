"""API a v handler module."""

from datetime import datetime, timedelta, timezone

import pandas

from app.alpaca.alpaca_handler import AlpacaAccountClient
from app.binance_handler.binance_handler import BinanceHandler
from app.yahoo.yahoo_handler import YahooHandler

from app.tickers.schemas import Stock, TickerData

from config.logger_config import logger


class TickerHandler:
    """_summary_"""

    def __init__(self, name: str) -> None:
        self.name: str = name

        self.data = TickerData(name=name)

        self.data.stock = Stock(stock=[])

        self.yahoo_handler = YahooHandler(ticker=self.name)

        self.binance_client = BinanceHandler(main_currency="USDC")

    async def download_data(self, interval: str) -> None:
        """_summary_"""

        stock = self.binance_client.get_historical_data(
            ticker=self.name, interval=interval
        )

        last_ts: pandas.Timestamp = stock.index[-1]

        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize(timezone.utc)

        else:
            last_ts = last_ts.astimezone(timezone.utc)

        now = datetime.now(timezone.utc)

        diff = now - last_ts

        threshold = timedelta(hours=1, minutes=10)

        if diff > threshold:
            logger.warning("TICKER HANDLER Data is older than threshold: %s", diff)

            self.data.stock = pandas.DataFrame()

            return

        self.data.stock = stock
