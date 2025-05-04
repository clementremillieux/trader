"""Test suite for BinanceHandler."""

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

info = binance_handler.get_symbol_lot_size(
    symbol="BTCUSDT",
)

print(info.model_dump_json(indent=2))

# stock = binance_handler.get_historical_data(symbol="BTCUSDT", interval="1h", days=150)

# print(stock.head())

# print(stock.tail())

# print("Found %s rows" % len(stock))

binance_handler.submit_order(
    symbol="ORCAUSDT",
    side="BUY",
    order_type="MARKET",
    quantity=5,
)
