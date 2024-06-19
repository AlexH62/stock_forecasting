from models.model import Model
from models.node import ODEFunc
from utils import sequence

import numpy as np
import tensorflow as tf
from keras import Model as KerasModel
from keras.api.layers import Layer, Dense, LeakyReLU
from keras.api.optimizers import Adam
from keras.api.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow_probability.python.math.ode import DormandPrince
    
class ODEBlock(KerasModel):
    def __init__(self, odefunc, augment_dim, tol=1e-3):
        super(ODEBlock, self).__init__()

        self.augment_dim = augment_dim
        self.odefunc = odefunc
        self.tol = tol
        self.solver = DormandPrince(max_num_steps=1000)

    def call(self, y0):
        aug = tf.zeros_like(y0)
        aug = aug[:, :, :self.augment_dim]
        y0_aug = tf.concat([y0, aug], axis=-1)
        out = self.solver.solve(
            self.odefunc, 
            initial_time=0.,
            initial_state=y0_aug,
            solution_times=[1.]
        )
        return out.states[-1]

class ODENet(KerasModel):
    def __init__(self, hidden_dim, lookback, output_dim, depth, augment_dim, tol=1e-3):
        super(ODENet, self).__init__()
        assert augment_dim < lookback
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tol = tol

        self.odeblocks = [ODEBlock(ODEFunc(hidden_dim, lookback + augment_dim), augment_dim=augment_dim, tol=tol) for _ in range(depth)]
        self.linear_layer = Dense(self.output_dim)

    def call(self, y0, training=None):
        x = y0
        for layer in self.odeblocks:
            x = layer(x)

        return self.linear_layer(x)

class ANODE(Model):
    def __init__(self):
        self.name = "ANODE"

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
            augment_dim=5,
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