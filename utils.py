import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def split(data, validation=True):
    split_idx = int(0.7 * len(data))
    val_idx = int(0.85 * len(data))

    if validation:
        return data[:split_idx], data[split_idx:val_idx], data[val_idx:]
    
    return data[:val_idx], data[val_idx:]

def sequence(data, lookback, horizon=1):
    x, y = [], []

    for i in range(len(data)):
        end_ix = i + lookback
        y_ix = end_ix + horizon - 1
        
        if y_ix > len(data)-1:
            break
        
        seq_x, seq_y = data[i:end_ix], data[y_ix]
        x.append(seq_x)
        y.append(seq_y)

    x, y = np.array(x), np.array(y)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    y = y.reshape((y.shape[0], 1))

    return x, y

def plot(y_train, y_test, y_hat, model_type, ticker, lookahead):
    x = np.linspace(0, len(y_train)-1, len(y_train))
    x_test = np.linspace(len(y_train)-1, len(y_train) + len(y_test)-1, len(y_test)+1)

    test_extended = np.append([y_train[-1]], y_test)
    hat_extended = np.append([y_train[-1]], y_hat)

    fig, ax = plt.subplots()

    ax.plot(x, y_train, label='Train & Val', color='grey', linewidth=1.2)
    ax.plot(x_test, test_extended, label='True', color='#518af5', linewidth=1.2)
    ax.plot(x_test, hat_extended, label='Predicted', color='#f58782', linewidth=1.)
    ax.grid()
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.title(model_type + ' ' + ticker)
    plt.savefig('figs/' + model_type + '/lookahead_' + str(lookahead) + '/' + ticker)
    plt.close()

def write_to_csv(filename, modelname, tickers, maes, mapes, rmses, r2s):
    path = Path(filename)
    if path.is_file():
      file = open(filename, "a")
    else:
      file = open(filename, "w")
    file.write(modelname + ",MAE,MAPE,RMSE,R2\n")
    for i, ticker in enumerate(tickers):
      file.write(ticker + "," + str(round(maes[i], 5)) + "," + str(round(mapes[i], 5)) + "," + str(round(rmses[i], 5)) + "," + str(round(r2s[i], 5)) + "\n")
    file.close()