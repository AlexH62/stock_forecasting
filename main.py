import repository
import preprocessor
import metrics

import random
import torch
import warnings
import keras
import numpy as np
from models.temp import Transformer
from models.gru import GRU
from models.lstm import LSTM
from models.ltc import LTC
#from models.transformer import Transformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

keras.utils.set_random_seed(42)
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

warnings.filterwarnings('ignore')

TICKER = ["^N225"]
#TICKER = ["NVDA", "IBM", "AAPL", "NFLX", "GOOG", "GS", "JPM", "BCS", "SAN", "MS"]
PERIOD = "1y"

rmses = []
maes = []
mapes = []
r2s = []

for ticker in TICKER:
    data = repository.get_data(ticker, PERIOD)
    scaler = RobustScaler()

    train, val, test = preprocessor.split(data)

    scaled_train = scaler.fit_transform(train.reshape(-1, 1))
    scaled_val = scaler.transform(val.reshape(-1, 1))
    scaled_test = scaler.transform(test.reshape(-1, 1))

    # Change model here
    model = Transformer()

    model.fit(scaled_train, val=scaled_val, neurons=1, epochs=1)

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