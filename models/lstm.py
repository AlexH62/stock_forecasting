from models.model import Model
from utils import sequence

import numpy as np
from keras import Sequential
from keras.api.layers import Dense, LSTM as KerasLSTM
from keras.api.callbacks import EarlyStopping
from keras.api.optimizers import Adam

class LSTM(Model):
    def __init__(self):
        self.name = "LSTM"

    def fit(self, train, val=None, depth=10, epochs=200, lookback=30, horizon=1):
        self.train = train
        self.lookback = lookback
        self.horizon = horizon
        if val is not None:
            self.val = val

        x, y = sequence(train, lookback, horizon=horizon)
        if val is not None:
            extended_val = np.append(self.train[-self.lookback-self.horizon+1:], val)
            x_val, y_val = sequence(extended_val, lookback, horizon=horizon)

        self.model = Sequential()
        for _ in range(depth-1):
            self.model.add(KerasLSTM(512, return_sequences=True))
        self.model.add(KerasLSTM(512))
        self.model.add(Dense(1))
        optimizer = Adam(learning_rate=1e-3)

        self.model.compile(optimizer=optimizer, loss='mse')

        if val is not None:
            early_stopping = EarlyStopping(
                                monitor='val_loss',
                                min_delta=0,
                                patience=20,
                                restore_best_weights=True
                            )
            self.model.fit(x, y, validation_data=(x_val, y_val), epochs=epochs, callbacks=[early_stopping])
        else:
            early_stopping = EarlyStopping(
                                monitor='val_loss',
                                min_delta=0,
                                patience=20,
                                restore_best_weights=True
                            )
            self.model.fit(x, y, epochs=epochs, callbacks=[early_stopping])
        
        self.model.summary()