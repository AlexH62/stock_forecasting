from models.model import Model
from preprocessor import sequence

import numpy as np
from ncps import wirings
from ncps.tf import LTC, CfC
from keras.models import Sequential
from keras.layers import InputLayer

class LTC(Model):
    def __init__(self):
        self.name = "LTC"

    def fit(self, train, neurons, epochs, lookback=30):
        self.train = train
        self.lookback = lookback
        
        x, y = sequence(train, lookback)
        
        wiring = wirings.AutoNCP(neurons, 1)

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(None, x.shape[2])))
        self.model.add(CfC(wiring))
        self.model.compile(optimizer="adam", loss='mse')

        self.model.fit(x, y, epochs=epochs)

    def predict(self, x):
        extended_data = np.append(self.train[-self.lookback:], x)
        inp, _ = sequence(extended_data, self.lookback, 1)
        return self.model.predict(inp)