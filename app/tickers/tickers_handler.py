"""API a v handler module."""

from app.yahoo.yahoo_handler import YahooHandler

from app.tickers.schemas import Stock, TickerData


class TickerHandler:
    """_summary_"""

    def __init__(self, name: str) -> None:
        self.name: str = name

        self.data = TickerData(name=name)

        self.data.stock = Stock(stock=[])

        self.yahoo_handler = YahooHandler(ticker=self.name)

    async def download_data(self, interval: str, days: str) -> None:
        """_summary_"""

        stock = self.yahoo_handler.get_stock_data(interval=interval, days=days)

        self.data.stock = stock
