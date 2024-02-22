from repository import Repository
from preprocessor import Preprocessor
from metrics import Metrics
from arch import arch_model
import pmdarima
import numpy as np

TICKER = ["NVDA", "IBM", "AAPL", "NFLX", "GOOG", "GS", "JPM", "BCS", "SAN", "MS"]
PERIOD = "1y"

rmses = []

for ticker in TICKER:
  repository = Repository()
  data = repository.get_data(ticker, PERIOD)

  STEPS = 30
  preprocessor = Preprocessor()
  _, y_train, _, y_test = preprocessor.sequence(data, STEPS)

  y_hat_all = []

  for i in range(len(y_test)):
    y_train_iterate = np.append(y_train, y_test[:i])
    arima_model_fitted = pmdarima.auto_arima(y_train_iterate)
    arima_residuals = arima_model_fitted.arima_res_.resid

    # fit a GARCH(1,1) model on the residuals of the ARIMA model
    garch = arch_model(arima_residuals, p=1, q=1)
    garch_fitted = garch.fit()

    # Use ARIMA to predict mu
    predicted_mu = arima_model_fitted.predict(n_periods=1)
    # Use GARCH to predict the residual
    garch_forecast = garch_fitted.forecast(horizon=1)
    predicted_et = garch_forecast.mean['h.1'].iloc[-1]
    # Combine both models' output: yt = mu + et
    prediction = predicted_mu + predicted_et

    y_hat_all = np.append(y_hat_all, prediction)
    print(len(y_test) - i)

  metrics = Metrics()
  rmse = metrics.print_RMSE(y_test, y_hat_all)
  metrics.plot(y_train, y_test, y_hat_all, "GARCH", ticker)
  rmses.append(rmse)

print(rmses)