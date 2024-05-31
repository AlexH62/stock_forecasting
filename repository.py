import yfinance
import numpy as np

def get_data(ticker, start='2021-01-01', end='2023-01-01', interval='1d'):
   info = yfinance.Ticker(ticker)

   # Valid periods are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, and max.
   data = info.history(start=start, end=end, interval=interval)
   data =  data['Close']
   return data.to_numpy()

# Useful utility to ensure models are working as expected
def generate_fake_data(length=300, mean=20, std_dev=2, trend=0.01, seasonality_amp=10, seasonality_period=50):
    time = np.arange(length)
    trend_component = trend * time
    seasonality_component = seasonality_amp * np.sin(2 * np.pi * time / seasonality_period)
    noise = np.random.normal(loc=mean, scale=std_dev, size=length)

    time_series = mean + trend_component + seasonality_component + noise
    
    return time_series