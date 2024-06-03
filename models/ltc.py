from models.model import Model
from preprocessor import sequence

import numpy as np
from ncps import wirings
from ncps.tf import LTC, CfC
from keras.models import Sequential
from keras.layers import InputLayer
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

class LTC(Model):
    def __init__(self):
        self.name = "LTC"

    def fit(self, train, val=None, depth=10, epochs=200, lookback=30):
        self.train = train
        self.val = val
        self.lookback = lookback
        
        x, y = sequence(train, lookback)
        if val is not None:
            extended_val = np.append(self.train[-self.lookback:], val)
            x_val, y_val = sequence(extended_val, lookback)
        
        wiring = wirings.AutoNCP(depth, 1)

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(None, x.shape[2])))
        self.model.add(CfC(wiring))

        optimizer = Adam(learning_rate=1e-3)

        self.model.compile(optimizer=optimizer, loss='mse')

        if val is not None:
            reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.1,
                            patience=5, 
                            min_lr=1e-9
                        )
            early_stopping = EarlyStopping(
                                monitor='val_loss',
                                min_delta=0,
                                patience=10
                            )
            self.model.fit(x, y, validation_data=(x_val, y_val), epochs=epochs, callbacks=[reduce_lr, early_stopping])
        else:
            reduce_lr = ReduceLROnPlateau(
                            monitor='loss',
                            factor=0.1,
                            patience=5, 
                            min_lr=1e-9
                        )
            early_stopping = EarlyStopping(
                                monitor='loss',
                                min_delta=0,
                                patience=10
                            )
            self.model.fit(x, y, epochs=epochs, callbacks=[reduce_lr, early_stopping])

    def predict(self, x):
        if hasattr(self, 'val'):
            extended_data = np.append(self.val[-self.lookback:], x)
        else:
            extended_data = np.append(self.train[-self.lookback:], x)
        inp, _ = sequence(extended_data, self.lookback, 1)
        return self.model.predict(inp)