from repository import Repository
from preprocessor import Preprocessor
from metrics import Metrics
import pmdarima
import numpy as np

TICKER = ["SAN"]#, "NVDA", "IBM", "AAPL", "NFLX", "GOOG", "GS", "JPM", "BCS", "SAN", "MS"]
PERIOD = "1y"

rmses = []
for ticker in TICKER:
  repository = Repository()
  data = repository.get_data(ticker, PERIOD)

  #STEPS = 30
  #LOOKAHEAD = 1

  preprocessor = Preprocessor()
  #_, y_train, _, y_test_clean = preprocessor.sequence(data, STEPS)
  #_, _, _, y_test = preprocessor.sequence(data, STEPS, lookahead=LOOKAHEAD)
  split = int(data.shape[0] * 0.7)
  train = data[:split]
  test = data[split:]

  y_hat_all = []

  for i in range(len(test)):
    y_train_iterate = np.append(train, test[:i])
    arima_model = pmdarima.auto_arima(y_train_iterate)
    y_hat = arima_model.predict(n_periods=1)[-1]

    y_hat_all = np.append(y_hat_all, y_hat)
    print(len(test) - i)

  metrics = Metrics()
  rmse = metrics.print_RMSE(test, y_hat_all)
  metrics.plot(train, test, y_hat_all, "ARIMA", ticker, 0)
  rmses.append(rmse)

print(rmses)
file = open("dump.txt", "a")
file.write("ARIMA \n")
for val in rmses:
	file.write(str(round(val, 4)) + "\n")
file.close()