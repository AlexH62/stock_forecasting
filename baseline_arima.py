import repository
import preprocessor
import metrics

import pmdarima
import numpy as np

TICKER = ["MS"]#, "NVDA", "IBM", "AAPL", "NFLX", "GOOG", "GS", "JPM", "BCS", "SAN", "MS"]
PERIOD = "1y"

rmses = []
for ticker in TICKER:
    data = repository.get_data(ticker, PERIOD)
    train, test = preprocessor.split(data)

    y_hat_all = []

    for i in range(len(test)):
        y_train_iterate = np.append(train, test[:i])
        arima_model = pmdarima.auto_arima(y_train_iterate)
        y_hat = arima_model.predict(n_periods=1)[-1]

        y_hat_all = np.append(y_hat_all, y_hat)
        print(len(test) - i)

    rmse = metrics.print_RMSE(test, y_hat_all)
    metrics.plot(train, test, y_hat_all, "ARIMA", ticker, 0)
    rmses.append(rmse)

print(rmses)
file = open("dump.txt", "a")
file.write("ARIMA \n")
for val in rmses:
    file.write(str(round(val, 4)) + "\n")
file.close()