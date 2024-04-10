import repository
import preprocessor
import metrics

import keras
from models.gru import GRU
from models.lstm import LSTM
from models.ltc import LTC
from models.transformer import Transformer
from sklearn.preprocessing import MinMaxScaler

keras.utils.set_random_seed(42)

TICKER = ["MS"]#"NVDA", "IBM", "AAPL", "NFLX", "GOOG", "GS", "JPM", "BCS", "SAN", "MS"]
PERIOD = "1y"

rmses = []
for ticker in TICKER:
    data = repository.get_data(ticker, PERIOD)
    scaler = MinMaxScaler()

    train, test = preprocessor.split(data)

    scaled_train = scaler.fit_transform(train.reshape(-1, 1))
    scaled_test = scaler.transform(test.reshape(-1, 1))

    model = LTC()
    model.fit(scaled_train, 10, 150)

    predictions = model.predict(scaled_test)

    unscaled_prediction = scaler.inverse_transform(predictions.reshape(-1, 1))

    rmse = metrics.print_RMSE(test, unscaled_prediction)
    metrics.plot(train, test, unscaled_prediction, model.name, ticker, 1)
    rmses.append(rmse)

print(rmses)
file = open("dump.txt", "a")
file.write(model.name + "\n")
for val in rmses:
    file.write(str(round(val, 4)) + "\n")
file.close()