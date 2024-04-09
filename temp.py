from repository import Repository
from metrics import Metrics
import matplotlib.pyplot as plt
from preprocessor import Preprocessor
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import random
import numpy as np

NEURONS = 20
EPOCHS = 60
random.seed(7)

repository = Repository()
data = repository.get_data("SAN", "1y")

preprocessor = Preprocessor()
scaled = preprocessor.scale(data)

split = int(scaled.shape[0] * 0.7)
train = scaled[:split]
test = scaled[split:]

def fit_lstm(train, epochs, neurons):
  batch_size = 1
  X = train[:-1]
  y = train[1:]
  X = np.reshape(X, (len(X), 1, 1))
  y = np.reshape(y, (len(y), 1, 1))

  model = Sequential()
  model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
  model.add(Dense(y.shape[1]))
  #optim = Adam(learning_rate=0.0001)
  model.compile(optimizer="adam", loss='mse')

  #rmses = []
  for i in range(epochs):
    #preds = []
    model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False)
    #for i in test:
    #  x = [i]
    #  x = np.reshape(x, (1, len(x), 1))
    #  yhat = model.predict(x)
    #  preds.append(yhat)
    #preds = np.array(preds)

    #unscaled_actual = preprocessor.reverse_transform(test)
    #unscaled_prediction = preprocessor.reverse_transform(preds)

    #metrics = Metrics()
    #rmse = metrics.print_RMSE(unscaled_actual, unscaled_prediction)
    #rmses.append(rmse)
    model.reset_states()
  
  #file = open("dump.txt", "a")
  #for i, val in enumerate(rmses):
  #  file.write(str(i) + ":" + str(round(val, 4)) + "\n")
  #file.close()

  return model

preds = []
lstm_model = fit_lstm(train, EPOCHS, NEURONS)
lstm_model.predict(np.reshape(train, (len(train), 1, 1)))

for i in test:
  x = [i]
  x = np.reshape(x, (1, len(x), 1))
  yhat = lstm_model.predict(x)
  preds.append(yhat)

preds = np.array(preds)

unscaled_train = preprocessor.reverse_transform(train)
unscaled_actual = preprocessor.reverse_transform(test)
unscaled_prediction = preprocessor.reverse_transform(preds)

metrics = Metrics()
metrics.print_RMSE(unscaled_actual, unscaled_prediction)

metrics.plot(unscaled_train, unscaled_actual, unscaled_prediction, "LSTM", "^N225", 0)