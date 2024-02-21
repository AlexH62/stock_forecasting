import yfinance as yahooFinance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.list_physical_devices('GPU')

def get_data(ticker, period="max"):
    info = yahooFinance.Ticker(ticker)

    # Valid periods are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, and max.
    return info.history(period=period)

ticker = "BCS"
data = get_data(ticker, period="1y")
data =  data["Close"]
data = data.to_numpy()

#scaler = MinMaxScaler(feature_range=(0, 1))

#scaled_data = scaler.fit_transform(data.reshape(-1, 1))

plt.plot(data)
plt.show()

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

n_steps = 30
# split into samples
X, y = split_sequence(data, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

scaler = MinMaxScaler(feature_range=(0, 1))

train_samples = int(0.8 * X.shape[0])
X_train = X[:train_samples]
y_train = y[:train_samples]

y_train = scaler.fit_transform(y_train.reshape(-1, 1)).squeeze()
X_train = scaler.transform(X_train.reshape(-1, 1)).reshape(-1, 30)

X_test = scaler.transform(X[train_samples:].reshape(-1, 1)).reshape(-1, 30)
y_test = scaler.transform(y[train_samples:].reshape(-1, 1)).squeeze()

print(X.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
#optim = Adam(learning_rate=0.0001)
model.compile(optimizer="adam", loss='mse')
model.fit(X_train, y_train, epochs=200, validation_split=0.1875, shuffle=False)

y_hat = model.predict(X_test)

rmse = np.sqrt(np.sum(np.square(y_test - y_hat.squeeze())) / len(y_test))
print(rmse)

x = np.linspace(0, len(y_train)-1, len(y_train))
x_test = np.linspace(len(y_train), len(y_train) + len(y_test)-1, len(y_test))

plt.plot(x, scaler.inverse_transform(y_train.reshape(-1, 1)))
plt.plot(x_test, scaler.inverse_transform(y_test.reshape(-1, 1)), label="true")
plt.plot(x_test, scaler.inverse_transform(y_hat.reshape(-1, 1)), label="predicted")
plt.legend(loc="upper left")
title = "LSTM " + ticker
plt.title(title)
plt.show()