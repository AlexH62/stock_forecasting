import repository
import utils
import metrics

import argparse
from arch import arch_model
import pmdarima
import numpy as np

TICKERS = ['NVDA', 'IBM', 'AAPL', 'NFLX', 'GOOG', 'GS', 'JPM', 'BCS', 'SAN', 'MS']
START = '2021-01-01'
END = '2023-01-01'
LOOKBACK = 60

parser = argparse.ArgumentParser("GARCH benchmark")
parser.add_argument("horizon", help="Forecasting horizon", type=int)
args = parser.parse_args()

HORIZON = args.horizon

rmses = []
maes = []
mapes = []
r2s = []

for ticker in TICKERS:
    data = repository.get_data(ticker, START, END)
    train, test = utils.split(data, validation=False)

    y_hat_all = []

    for i in range(len(test)):
        idx = len(test) + HORIZON - 1
        arima_model_fitted = pmdarima.auto_arima(data[-idx + i - LOOKBACK:-idx + i])
        arima_residuals = arima_model_fitted.arima_res_.resid

        garch = arch_model(arima_residuals, p=1, q=1)
        garch_fitted = garch.fit(disp=False)

        mu = arima_model_fitted.predict(n_periods=HORIZON)[-1]
        garch_forecast = garch_fitted.forecast(horizon=HORIZON)
        et = garch_forecast.mean[f'h.{HORIZON}'].iloc[-1]
        prediction = mu + et

        y_hat_all = np.append(y_hat_all, prediction)
        print(f'Forecasts completed: {i+1}/{len(test)}')

    print(f'Metrics for {ticker}')
    rmse, mae, mape, r2 = metrics.print_all(test, y_hat_all)
    
    rmses.append(rmse)
    maes.append(mae)
    mapes.append(mape)
    r2s.append(r2)
    
    utils.plot(train, test, y_hat_all, 'GARCH', ticker, HORIZON)

utils.write_to_csv('results.csv', 'GARCH', TICKERS, maes, mapes, rmses, r2s)