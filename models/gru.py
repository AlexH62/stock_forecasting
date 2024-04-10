from models.model import Model

import numpy as np
from keras.models import Sequential
import keras.layers

class GRU(Model):
    def __init__(self):
        self.name = "GRU"

    def fit(self, train, neurons, epochs):
        batch_size = 1
        self.train = train

        x = train[:-1]
        y = train[1:]
        x = np.reshape(x, (len(x), 1, 1))
        y = np.reshape(y, (len(y), 1, 1))

        self.model = Sequential()
        self.model.add(keras.layers.GRU(neurons, batch_input_shape=(batch_size, x.shape[1], x.shape[2]), stateful=True))
        self.model.add(keras.layers.Dense(y.shape[1]))
        self.model.compile(optimizer="adam", loss='mse')

        for i in range(epochs):
            self.model.fit(x, y, epochs=1, batch_size=batch_size, shuffle=False)
            self.model.reset_states()
    
    def predict(self, data):
        # Check training has been done
        assert self.train is not None

        # Initialise state with training data
        preds = []
        self.model.predict(np.reshape(self.train, (len(self.train), 1, 1)))

        # Predict stepwise
        for i in data:
            x = [i]
            x = np.reshape(x, (1, len(x), 1))
            yhat = self.model.predict(x)
            preds.append(yhat)

        return np.array(preds)