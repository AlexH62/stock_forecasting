import yfinance

def get_data(ticker, period="max"):
   info = yfinance.Ticker(ticker)

   # Valid periods are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, and max.
   data = info.history(period=period)
   data =  data["Close"]
   return data.to_numpy()
