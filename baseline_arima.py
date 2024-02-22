from repository import Repository
from preprocessor import Preprocessor
from metrics import Metrics
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
    arima_model = pmdarima.auto_arima(y_train_iterate)
    y_hat = arima_model.predict(n_periods=1)

    y_hat_all = np.append(y_hat_all, y_hat)
    print(len(y_test) - i)

  metrics = Metrics()
  rmse = metrics.print_RMSE(y_test, y_hat_all)
  metrics.plot(y_train, y_test, y_hat_all, "ARIMA", ticker)
  rmses.append(rmse)

print(rmses)