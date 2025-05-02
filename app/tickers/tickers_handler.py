"""API a v handler module."""

from datetime import datetime, timedelta, timezone

import pandas

from app.alpaca.alpaca_handler import AlpacaAccountClient
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

        self.alpaca_client = AlpacaAccountClient(
            api_key="PKPAPAB1JWAA5PLJSFZ4",
            secret_key="HTMo7Lj68OpsXsiTkKIn6JV4FYlWtNtqEbd22vlv",
            paper=True,
        )

    async def download_data(self, interval: str, days: str) -> None:
        """_summary_"""

        # stock = self.yahoo_handler.get_stock_data(interval=interval, days=days)

        stock = self.alpaca_client.get_ticker_data(
            ticker=self.name,
            interval=interval,
            days=days,
        )

        last_ts: pandas.Timestamp = stock.index[-1]

        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize(timezone.utc)

        else:
            last_ts = last_ts.astimezone(timezone.utc)

        now = datetime.now(timezone.utc)

        diff = now - last_ts

        threshold = timedelta(hours=1, minutes=1)

        if diff > threshold:
            logger.warning("TICKER HANDLER Data is older than threshold: %s", diff)

            self.data.stock = pandas.DataFrame()

            return

        self.data.stock = stock
