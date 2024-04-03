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
  LOOKAHEAD = 20

  preprocessor = Preprocessor()
  _, y_train, _, y_test_clean = preprocessor.sequence(data, STEPS)
  _, _, _, y_test = preprocessor.sequence(data, STEPS, lookahead=LOOKAHEAD)

  y_hat_all = []

  for i in range(len(y_test)):
    y_train_iterate = np.append(y_train, y_test_clean[:i])
    arima_model = pmdarima.auto_arima(y_train_iterate)
    y_hat = arima_model.predict(n_periods=LOOKAHEAD)[-1]

    y_hat_all = np.append(y_hat_all, y_hat)
    print(len(y_test) - i)

  metrics = Metrics()
  rmse = metrics.print_RMSE(y_test, y_hat_all)
  metrics.plot(y_train, y_test, y_hat_all, "ARIMA", ticker, LOOKAHEAD)
  rmses.append(rmse)

print(rmses)
file = open("dump.txt", "a")
file.write("ARIMA \n")
for val in rmses:
	file.write(str(round(val, 4)) + "\n")
file.close()