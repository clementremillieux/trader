"""Test suite for BinanceHandler."""

from typing import Any, Dict, List

from app.trader.schemas import PortfolioValue, Position

from app.binance_handler.binance_handler import BinanceHandler

binance_handler = BinanceHandler(main_currency="USDC")

exchange_info: Dict[str, Any] = binance_handler.client.exchange_info()

symbols = exchange_info.get("symbols", [])

trading_pairs = [s for s in symbols if s.get("status") == "TRADING"]

print(trading_pairs)
tickers: List[str] = [s["symbol"] for s in trading_pairs]


print(tickers)

print(len(tickers))

tickers_not_in_usd = [ticker for ticker in tickers if not ticker.endswith("USDT")]


print(len(tickers_not_in_usd))

# portfolio: PortfolioValue = binance_handler.get_portfolio()

# print(portfolio.model_dump_json(indent=2))

# positions: List[Position] = binance_handler.get_positions()

# for position in positions:
#     print(position.model_dump_json(indent=2))

# binance_handler.get_all_tickers()

# stock = binance_handler.get_historical_data(ticker="BTC", interval="1h")

# print(stock.head())

# print(stock.tail())

# print("Found %s rows" % len(stock))

# binance_handler.submit_order(
#     symbol="ORCA",
#     side="BUY",
#     order_type="MARKET",
#     quantity=5,
# )
