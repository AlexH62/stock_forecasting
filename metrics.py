import numpy as np
import matplotlib.pyplot as plt

def print_RMSE(actual, predicted):
    rmse = np.sqrt(np.sum(np.square(actual - predicted)) / len(actual))
    print(rmse)
    return rmse

def print_MAE(actual, predicted):
    mae = np.sum(np.abs(actual - predicted)) / len(actual)
    print(mae)
    return mae

def print_MAPE(actual, predicted):
    mape = np.sum(np.abs(np.divide(actual - predicted, actual))) / len(actual)
    print(mape)
    return mape

def print_R2(actual, predicted):
    num = np.sum(np.square(actual - predicted))
    denom = np.sum(np.square(actual - np.mean(actual)))
    r2 = 1 - (num / denom)
    print(r2)
    return r2

def plot(y_train, y_test, y_hat, model_type, ticker, lookahead):
    x = np.linspace(0, len(y_train)-1, len(y_train))
    x_test = np.linspace(len(y_train)-1, len(y_train) + len(y_test)-1, len(y_test)+1)

    test_extended = np.append([y_train[-1]], y_test)
    hat_extended = np.append([y_train[-1]], y_hat)

    plt.plot(x, y_train)
    plt.plot(x_test, test_extended, label="true")
    plt.plot(x_test, hat_extended, label="predicted")
    plt.legend(loc="upper left")
    plt.title(model_type + " " + ticker)
    plt.savefig("figs/" + model_type + "/lookahead_" + str(lookahead) + "/" + ticker)
    plt.close()