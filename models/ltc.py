from models.model import Model
from preprocessor import sequence

import numpy as np
from ncps import wirings
from ncps.tf import LTC, CfC
from keras import Sequential
from keras import layers

class LTC(Model):
    def __init__(self):
        self.name = "LTC"

    def fit(self, train, val=None, neurons=10, epochs=200, lookback=30):
        self.train = train
        self.val = val
        self.lookback = lookback
        
        x, y = sequence(train, lookback)
        if val is not None:
            extended_val = np.append(self.train[-self.lookback:], val)
            x_val, y_val = sequence(extended_val, lookback)
        
        wiring = wirings.AutoNCP(neurons, 1)

        self.model = Sequential()
        self.model.add(layers.InputLayer(input_shape=(None, x.shape[2])))
        self.model.add(CfC(wiring))
        self.model.compile(optimizer="adam", loss='mse')

        if val is not None:
            self.model.fit(x, y, validation_data=(x_val, y_val), epochs=epochs)
        else:
            self.model.fit(x, y, epochs=epochs)

    def predict(self, x):
        if hasattr(self, 'val'):
            extended_data = np.append(self.val[-self.lookback:], x)
        else:
            extended_data = np.append(self.train[-self.lookback:], x)
        inp, _ = sequence(extended_data, self.lookback, 1)
        return self.model.predict(inp)