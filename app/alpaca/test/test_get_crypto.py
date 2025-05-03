from yahooquery import Screener

s = Screener()
tickers = []


data = s.get_screeners("all_cryptocurrencies_us", count=250)

quotes = data.get("all_cryptocurrencies_us", {}).get("quotes", [])

tickers.extend([item["symbol"] for item in quotes])

tickers = list(set(tickers))

print(tickers)

print("Number of tickers: ", len(tickers))
