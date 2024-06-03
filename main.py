import repository
import preprocessor
import metrics

import random
import warnings
import keras
import tensorflow as tf
import numpy as np
from models.gru import GRU # 2
from models.lstm import LSTM # 5
from models.ltc import LTC # 50, 100
from models.transformer import Transformer # 3
from models.node import NODE # 1
from sklearn.preprocessing import RobustScaler

keras.utils.set_random_seed(42)
random.seed(42)
np.random.seed(42)

class Scaler():
    def __init__(self):
        pass

    def fit_transform(self, data):
        return data
    def transform(self, data):
        return data
    def inverse_transform(self, data):
        return data

#TICKER = ["^N225"]
TICKER = ["NVDA", "IBM", "AAPL", "NFLX", "GOOG", "GS", "JPM", "BCS", "SAN", "MS"]
START = '2021-01-01'
END ='2023-01-01'

rmses = []
maes = []
mapes = []
r2s = []

for ticker in TICKER:
    data = repository.get_data(ticker, START, END)
    #data = repository.generate_fake_data()
    scaler = RobustScaler()

    train, val, test = preprocessor.split(data)

    scaled_train = scaler.fit_transform(train.reshape(-1, 1))
    scaled_val = scaler.transform(val.reshape(-1, 1))
    scaled_test = scaler.transform(test.reshape(-1, 1))

    # Change model here
    model = Transformer()

    model.fit(scaled_train, val=scaled_val, depth=3, lookback=30, epochs=200)

    predictions = model.predict(scaled_test)

    unscaled_prediction = scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()

    rmse = metrics.print_RMSE(test, unscaled_prediction)
    mae = metrics.print_MAE(test, unscaled_prediction)
    mape = metrics.print_MAPE(test, unscaled_prediction)
    r2 = metrics.print_R2(test, unscaled_prediction)

    rmses.append(rmse)
    maes.append(mae)
    mapes.append(mape)
    r2s.append(r2)

    train_val = np.append(train, val)
    metrics.plot(train_val, test, unscaled_prediction, model.name, ticker, 1)

file = open("dump.csv", "a")
file.write(model.name + ",MAE,MAPE,RMSE,R2\n")
for i, ticker in enumerate(TICKER):
    file.write(ticker + "," + str(round(maes[i], 4)) + "," + str(round(mapes[i], 4)) + "," + str(round(rmses[i], 4)) + "," + str(round(r2s[i], 4)) + "\n")
file.close()