from models.model import Model

import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, GRU as KerasGRU
from keras.optimizers import Adam

class GRU(Model):
    def __init__(self):
        self.name = "GRU"

    def fit(self, train, val=None, depth=10, epochs=200, lookback=30):
        batch_size = 1
        self.train = train

        x = train[:-1]
        y = train[1:]
        x = np.reshape(x, (len(x), 1, 1))
        y = np.reshape(y, (len(y), 1, 1))

        if val is not None:
            self.val = val
            x_val = val[:-1]
            y_val = val[1:]
            x_val = np.reshape(x_val, (len(x_val), 1, 1))
            y_val = np.reshape(y_val, (len(y_val), 1, 1))

        self.model = Sequential()
        self.model.add(KerasGRU(depth, batch_input_shape=(batch_size, x.shape[1], x.shape[2]), stateful=True))
        self.model.add(Dense(y.shape[1]))

        optimizer = Adam(learning_rate=1e-3)

        self.model.compile(optimizer=optimizer, loss='mse')

        early_stoping_counter = 0
        lowest_loss = sys.maxsize
        if val is not None:
            for i in range(epochs):
                print(f'Epoch {i+1}/{epochs}')
                history = self.model.fit(x, y, validation_data=(x_val, y_val), epochs=1, batch_size=batch_size, shuffle=False)
                val_loss = history.history['val_loss'][0]
                self.model.reset_states()
                if val_loss < lowest_loss:
                    lowest_loss = val_loss
                    early_stoping_counter = 0
                else:
                    early_stoping_counter += 1
                if early_stoping_counter == 20:
                    print('Early stopping condition met')
                    break
        else:
            for i in range(epochs):
                print(f'Epoch {i+1}/{epochs}')
                self.model.fit(x, y, epochs=1, batch_size=batch_size, shuffle=False)
                loss = history.history['loss'][0]
                self.model.reset_states()
                if loss <= lowest_loss:
                    lowest_loss = loss
                    early_stoping_counter = 0
                else:
                    early_stoping_counter += 1
                if early_stoping_counter == 10:
                    print('Early stopping condition met')
                    break
    
    def predict(self, data):
        # Check training has been done
        assert self.train is not None

        # Initialise state with training data
        preds = []
        self.model.predict(np.reshape(self.train, (len(self.train), 1, 1)), batch_size=1)

        if hasattr(self, 'val'):
            self.model.predict(np.reshape(self.val, (len(self.val), 1, 1)), batch_size=1)

        # Predict stepwise
        for i in data:
            x = [i]
            x = np.reshape(x, (1, len(x), 1))
            yhat = self.model.predict(x)
            preds.append(yhat)

        return np.array(preds)