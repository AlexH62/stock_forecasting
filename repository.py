import yfinance

def get_data(ticker, period="max"):
   info = yfinance.Ticker(ticker)

   # Valid periods are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, and max.
   data = info.history(start="2022-01-01", end="2023-01-01", interval="1d")
   data =  data["Close"]
   return data.to_numpy()


def get_data_temp(ticker, period="max"):
   info = yfinance.Ticker(ticker)

   # Valid periods are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, and max.
   data = info.history(start="2022-01-01", end="2023-01-01", interval="1d")
   return data