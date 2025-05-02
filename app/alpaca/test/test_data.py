from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

import pandas

# no keys required for crypto data
client = StockHistoricalDataClient(
    api_key="PK88IZ4CR3GJ7KBQ1INL",
    secret_key="XhpKvWsXeF1ki5m58dcXH6rsNv3Gwoqsuac4rMAd",
)

request_params = StockBarsRequest(
    symbol_or_symbols=["KGEI"],
    timeframe=TimeFrame.Hour,
    start=datetime(2024, 1, 1),
)

bars = client.get_stock_bars(request_params)

# convert to dataframe
df_wdc = bars.df.loc["KGEI"]

df_wdc.index = pandas.to_datetime(df_wdc.index).tz_convert("UTC")

print(df_wdc)

# # access bars as list - important to note that you must access by symbol key
# # even for a single symbol request - models are agnostic to number of symbols
# print(bars["WDC"])
