import repository
import preprocessor
import metrics

from arch import arch_model
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
        arima_model_fitted = pmdarima.auto_arima(y_train_iterate)
        arima_residuals = arima_model_fitted.arima_res_.resid

        # fit a GARCH(1,1) model on the residuals of the ARIMA model
        garch = arch_model(arima_residuals, p=1, q=1)
        garch_fitted = garch.fit(disp=False)

        # Use ARIMA to predict mu
        predicted_mu = arima_model_fitted.predict(n_periods=1)[-1]
        # Use GARCH to predict the residual
        garch_forecast = garch_fitted.forecast(horizon=1)
        predicted_et = garch_forecast.mean['h.1'].iloc[-1]
        # Combine both models' output: yt = mu + et
        prediction = predicted_mu + predicted_et

        y_hat_all = np.append(y_hat_all, prediction)
        print(len(test) - i)

    rmse = metrics.print_RMSE(test, y_hat_all)
    mae = metrics.print_MAE(test, y_hat_all)
    mape = metrics.print_MAPE(test, y_hat_all)
    r2 = metrics.print_R2(test, y_hat_all)

    rmses.append(rmse)
    maes.append(mae)
    mapes.append(mape)
    r2s.append(r2)
    
    metrics.plot(train, test, y_hat_all, "GARCH", ticker, 10)

file = open("dump.csv", "a")
file.write("GARCH,MAE,MAPE,RMSE,R2\n")
for i, ticker in enumerate(TICKER):
    file.write(ticker + "," + str(round(maes[i], 4)) + "," + str(round(mapes[i], 4)) + "," + str(round(rmses[i], 4)) + "," + str(round(r2s[i], 4)) + "\n")
file.close()