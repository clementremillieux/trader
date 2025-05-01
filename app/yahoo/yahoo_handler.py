"""YahooHandler module."""

import numpy as np

from datetime import datetime, timedelta

from typing import Dict

import pandas as pd

import yfinance as yf


class YahooHandler:
    """Class to handle Yahoo Finance API."""

    def __init__(self, ticker: str):
        self.ticker = ticker

        self.stock = yf.Ticker(ticker)

    def get_stock_data(self, interval: str, days: str) -> pd.DataFrame:
        """Function to get stock data from Yahoo Finance API."""

        start_date = datetime.now() - timedelta(days=int(days))

        start_time = start_date.strftime("%Y-%m-%d")

        end_date = datetime.now()

        end_time = end_date.strftime("%Y-%m-%d")

        df = self.stock.history(interval=interval, start=start_time, end=end_time)

        return df

    @staticmethod
    def get_static_features(info: Dict) -> None:
        """Function to get static features from Yahoo Finance API."""

        sector_str = info.get("sector", "Unknown")

        sector_hash = hash(sector_str)

        RANDOM_SEED = 10**8

        sector_numeric = (abs(sector_hash) % RANDOM_SEED) / RANDOM_SEED

        market_cap = float(info.get("marketCap", 0.0))

        book_value = float(info.get("bookValue", 0.0))

        price_to_book = float(info.get("priceToBook", 0.0))

        trailing_pe = float(info.get("trailingPE", 0.0))

        forward_pe = float(info.get("forwardPE", 0.0))

        revenue_growth = float(info.get("revenueGrowth", 0.0))

        earnings_growth = float(info.get("earningsGrowth", 0.0))

        gross_margins = float(info.get("grossMargins", 0.0))

        operating_margins = float(info.get("operatingMargins", 0.0))

        log_market_cap = np.log(market_cap + 1)

        static_features = np.array(
            [
                sector_numeric,
                log_market_cap,
                book_value,
                price_to_book,
                trailing_pe,
                forward_pe,
                revenue_growth,
                earnings_growth,
                gross_margins,
                operating_margins,
            ],
            dtype=float,
        )

        return static_features
