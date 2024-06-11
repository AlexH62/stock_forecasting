from models.model import Model
from utils import sequence

import numpy as np
from keras import Model as KerasModel
from keras.api.layers import Layer, Dense, LeakyReLU
from keras.api.optimizers import Adam
from keras.api.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow_probability.python.math.ode import DormandPrince, BDF

class ODEFunc(Layer):
    def __init__(self, hidden_dim, lookback):
        super(ODEFunc, self).__init__()
        self.dense1 = Dense(hidden_dim)
        self.dense2 = Dense(hidden_dim)
        self.dense3 = Dense(lookback)
        self.activation = LeakyReLU()

    # t in signature here required for solver
    def call(self, t, y):
        y = self.dense1(y)
        y = self.activation(y)
        y = self.dense2(y)
        y = self.activation(y)
        return self.dense3(y)
    
class ODEBlock(KerasModel):
    def __init__(self, odefunc, tol=1e-3):
        super(ODEBlock, self).__init__()

        self.odefunc = odefunc
        self.tol = tol
        self.solver = DormandPrince()

    def call(self, y0):
        out = self.solver.solve(
            self.odefunc, 
            initial_time=0.,
            initial_state=y0,
            solution_times=[1.]
        )
        return out.states[-1]

class ODENet(KerasModel):
    def __init__(self, hidden_dim, lookback, output_dim, depth, tol=1e-3):
        super(ODENet, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tol = tol

        self.odeblocks = [ODEBlock(ODEFunc(hidden_dim, lookback), tol=tol) for _ in range(depth)]
        self.linear_layer = Dense(self.output_dim)

    def call(self, y0, training=None):
        x = y0
        for layer in self.odeblocks:
            x = layer(x)

        return self.linear_layer(x)

class NODE(Model):
    def __init__(self):
        self.name = "NODE"

    def fit(self, train, val=None, depth=10, epochs=200, lookback=30, horizon=1):
        self.train = train
        self.lookback = lookback
        self.horizon = horizon
        
        x, y = sequence(train, lookback, horizon=horizon)
        x = np.expand_dims(x.squeeze(), axis=1)
        y = np.expand_dims(y, axis=1)
        if val is not None:
            self.val = val
            extended_val = np.append(self.train[-self.lookback-self.horizon+1:], val)
            x_val, y_val = sequence(extended_val, lookback, horizon=horizon)
            x_val = np.expand_dims(x_val.squeeze(), axis=1)
            y_val = np.expand_dims(y_val, axis=1)

        self.model = ODENet(
            hidden_dim=1024,
            depth=depth,
            lookback=lookback,
            output_dim=1
        )

        optimizer = Adam(learning_rate=1e-3)

        self.model.compile(optimizer=optimizer, loss='mse')

        if val is not None:
            reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=5,
                            min_lr=1e-9
                        )
            early_stopping = EarlyStopping(
                                monitor='val_loss',
                                min_delta=0,
                                patience=20,
                                restore_best_weights=True
                            )
            self.model.fit(x, y, validation_data=(x_val, y_val), epochs=epochs, callbacks=[reduce_lr, early_stopping])
        else:
            reduce_lr = ReduceLROnPlateau(
                            monitor='loss',
                            factor=0.5,
                            patience=5,
                            min_lr=1e-9
                        )
            early_stopping = EarlyStopping(
                                monitor='loss',
                                min_delta=0,
                                patience=20,
                                restore_best_weights=True
                            )
            self.model.fit(x, y, epochs=epochs, callbacks=[reduce_lr, early_stopping])
        
        self.model.summary()

    def predict(self, x):
        if hasattr(self, 'val'):
            pre_data = np.append(self.train, self.val)
        else:
            pre_data = self.train
        extended_data = np.append(pre_data[-self.lookback-self.horizon+1:], x)
        inp, _ = sequence(extended_data, self.lookback, self.horizon)
        inp = np.expand_dims(inp.squeeze(), axis=1)
        
        return self.model(inp, training=False).numpy()