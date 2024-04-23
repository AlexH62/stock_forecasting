import repository
import preprocessor
import metrics

import pmdarima
import numpy as np

#TICKER = ["^N225"]
TICKER = ["NVDA", "IBM", "AAPL", "NFLX", "GOOG", "GS", "JPM", "BCS", "SAN", "MS"]
PERIOD = "1y"

rmses = []
maes = []
mapes = []
r2s = []

for ticker in TICKER:
    data = repository.get_data(ticker, PERIOD)
    train, test = preprocessor.split(data, validation=False)

    y_hat_all = []

    for i in range(len(test)):
        y_train_iterate = np.append(train, test[:i])
        arima_model = pmdarima.auto_arima(y_train_iterate)
        y_hat = arima_model.predict(n_periods=1)[-1]

        y_hat_all = np.append(y_hat_all, y_hat)
        print(len(test) - i)

    rmse = metrics.print_RMSE(test, y_hat_all)
    mae = metrics.print_MAE(test, y_hat_all)
    mape = metrics.print_MAPE(test, y_hat_all)
    r2 = metrics.print_R2(test, y_hat_all)

    rmses.append(rmse)
    maes.append(mae)
    mapes.append(mape)
    r2s.append(r2)

    metrics.plot(train, test, y_hat_all, "ARIMA", ticker, 10)

file = open("dump.csv", "a")
file.write("ARIMA,MAE,MAPE,RMSE,R2\n")
for i, ticker in enumerate(TICKER):
    file.write(ticker + "," + str(round(maes[i], 4)) + "," + str(round(mapes[i], 4)) + "," + str(round(rmses[i], 4)) + "," + str(round(r2s[i], 4)) + "\n")
file.close()