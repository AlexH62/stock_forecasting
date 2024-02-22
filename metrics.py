import numpy as np
import matplotlib.pyplot as plt

class Metrics:
  def __init__(self):
    pass

  def print_RMSE(self, actual, predicted):
    rmse = np.sqrt(np.sum(np.square(actual - predicted)) / len(actual))
    print(rmse)
    return rmse

  def plot(self, y_train, y_test, y_hat, model_type, ticker):
    x = np.linspace(0, len(y_train)-1, len(y_train))
    x_test = np.linspace(len(y_train), len(y_train) + len(y_test)-1, len(y_test))

    plt.plot(x, y_train)
    plt.plot(x_test, y_test, label="true")
    plt.plot(x_test, y_hat, label="predicted")
    plt.legend(loc="upper left")
    plt.title(model_type + " " + ticker)
    plt.savefig("figs/" + model_type + "/" + ticker)
    plt.close()