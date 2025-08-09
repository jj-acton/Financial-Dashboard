import yfinance as yf

print(yf.__file__)
ticker = yf.Ticker("AAPL")
print(ticker.history(period="1d"))


