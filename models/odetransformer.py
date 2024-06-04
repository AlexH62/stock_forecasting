from models.model import Model
from models.anode import ODEBlock
from preprocessor import sequence

import numpy as np
from keras import Model as KerasModel
from keras.layers import Layer, MultiHeadAttention, Conv1D, LayerNormalization, Dropout, Dense, GlobalAveragePooling1D
from keras.activations import leaky_relu
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow_probability.python.math.ode import DormandPrince

class TransformerODEFunc(Layer):
    def __init__(self, heads=4, key_dim=256, lookback=30, dropout=0.05):
        super(TransformerODEFunc, self).__init__()
        self.attn = MultiHeadAttention(heads, key_dim, dropout=dropout)
        self.attn_norm = LayerNormalization(epsilon=1e-6)
        self.ff = Conv1D(filters=lookback, kernel_size=1, activation=leaky_relu)

    # t in signature here required for solver
    def call(self, t, y, training=None):
        x = self.attn_norm(y)
        x = self.attn(query=x, key=x, value=x, training=training)
        x = self.ff(x)

        return x
    
class ODENet(KerasModel):
    def __init__(self, lookback, output_dim, depth, augment_dim=0, tol=1e-3):
        super(ODENet, self).__init__()

        self.output_dim = output_dim
        self.tol = tol

        self.odeblocks = [ODEBlock(TransformerODEFunc(lookback=lookback + augment_dim), augment_dim=augment_dim, tol=tol) for _ in range(depth)]
        self.linear_layer = Dense(self.output_dim)

    def call(self, y0, training=None):
        x = y0
        for layer in self.odeblocks:
            x = layer(x)

        return self.linear_layer(x)
    
class ODETransformer(Model):
    def __init__(self):
        self.name = "ODETransformer"

    def fit(self, train, val=None, depth=10, epochs=200, lookback=30):
        self.train = train
        self.val = val
        self.lookback = lookback
        
        x, y = sequence(train, lookback)
        x = np.expand_dims(x.squeeze(), axis=1)
        y = np.expand_dims(y, axis=1)
        if val is not None:
            extended_val = np.append(self.train[-self.lookback:], val)
            x_val, y_val = sequence(extended_val, lookback)
            x_val = np.expand_dims(x_val.squeeze(), axis=1)
            y_val = np.expand_dims(y_val, axis=1)

        self.model = ODENet(
            depth=depth,
            augment_dim=0,
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
                                patience=20
                            )
            self.model.fit(x, y, epochs=epochs, callbacks=[reduce_lr, early_stopping])

    def predict(self, data):
        if self.val is not None:
            extended_data = np.append(self.val[-self.lookback:], data)
        else:
            extended_data = np.append(self.train[-self.lookback:], data)
        inp, _ = sequence(extended_data, self.lookback, 1)
        inp = np.expand_dims(inp.squeeze(), axis=1)
        
        return self.model(inp, training=False).numpy()