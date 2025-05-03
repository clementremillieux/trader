"""Test suite for BinanceHandler."""

import datetime
from typing import List

from app.trader.schemas import PortfolioValue, Position

from app.binance_handler.binance_handler import BinanceHandler

binance_handler = BinanceHandler()

portfolio: PortfolioValue = binance_handler.get_portfolio()

print(portfolio.model_dump_json(indent=2))

positions: List[Position] = binance_handler.get_positions()

for position in positions:
    print(position.model_dump_json(indent=2))

tickers: List[str] = binance_handler.get_all_tickers()


# stock = binance_handler.get_historical_data(symbol="BTCUSDT", interval="1h", days=150)

# print(stock.head())

# print(stock.tail())

# print("Found %s rows" % len(stock))

binance_handler.submit_order(
    symbol="BTCEUR",
    side="SELL",
    order_type="MARKET",
    quantity=0.00012,
)
