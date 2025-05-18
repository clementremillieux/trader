"""Test suite for BinanceHandler."""

from datetime import date

from app.binance_handler.binance_handler import BinanceHandler

binance_handler = BinanceHandler(main_currency="USDC")

interval = "15m"

stock = binance_handler.client.klines(
    symbol="BTCUSDC",
    interval=interval,
    limit=1000,
)


print("Found %s rows" % len(stock))

print(
    "From %s to %s"
    % (date.fromtimestamp(stock[0][0] / 1000), date.fromtimestamp(stock[-1][0] / 1000))
)

full_stock = stock

while True:
    stock = binance_handler.client.klines(
        symbol="BTCUSDC",
        interval=interval,
        limit=1000,
        startTime=stock[0][0] - 1000 * 60 * 60 * 24,
    )

    full_stock = stock + full_stock

    print(
        "From %s to %s [total stock %s]"
        % (
            date.fromtimestamp(stock[0][0] / 1000),
            date.fromtimestamp(stock[-1][0] / 1000),
            len(full_stock) + len(stock),
        )
    )

    if len(stock) < 1000:
        break

print("Found %s rows" % len(full_stock))
