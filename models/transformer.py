from models.model import Model
from preprocessor import sequence

import numpy as np
from keras.backend import bias_add
from keras.layers import Layer, MultiHeadAttention, Conv1D, LayerNormalization, Dropout, Dense, GlobalAveragePooling1D
from keras.activations import leaky_relu
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, Model as KerasModel

class PositionalEmbedding(Layer):
    def __init__(self, length):
        super(PositionalEmbedding, self).__init__()
        self.length = length

    def build(self, inputs_shape):
        feature_size = inputs_shape[-1]
        self.position_embeddings = self.add_weight(
            shape=[self.length, feature_size],
            initializer='glorot_uniform',
            trainable=True,
        )
        self.built = True
    
    def call(self, inputs):
        return bias_add(inputs, self.position_embeddings)

class TransformerEncoder(Layer):
    def __init__(self, heads, key_dim, ff_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.attn = MultiHeadAttention(heads, key_dim, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_norm = LayerNormalization(epsilon=1e-6)
        self.ff1 = Conv1D(filters=ff_dim, kernel_size=1, activation=leaky_relu)
        self.ff2 = Conv1D(filters=ff_dim, kernel_size=1, activation=leaky_relu)
        self.ff_dropout = Dropout(dropout)
        self.ff_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training):
        residual = x = inputs
        x = self.attn_norm(x)
        x = self.attn(query=x, key=x, value=x, training=training)
        x = self.attn_dropout(x, training=training)
        x = x + residual
        
        residual = x
        x = self.ff_norm(x)
        x = self.ff1(x)
        x = self.ff_dropout(x, training=training)
        x = self.ff2(x)

        return x + residual
    
class KerasTransformer(KerasModel):
    def __init__(self, sequence_length, enc_layers, heads, key_dim, ff_dim, mlp_units, dropout=0.05):
        super(KerasTransformer, self).__init__()
        self.dropout = dropout
        self.pos_emb = PositionalEmbedding(sequence_length)
        self.encoders = [
            TransformerEncoder(
                heads=heads,
                key_dim=key_dim,
                ff_dim=ff_dim,
                dropout=dropout,
            ) for _ in range(enc_layers)
        ]
        self.ffs = [
            Dense(dim) for dim in mlp_units
        ]
        self.pool = GlobalAveragePooling1D(data_format="channels_first")
        self.output_ff = Dense(1)

    def call(self, inputs):
        encoded =  self.pos_emb(inputs)

        for layer in self.encoders:
            encoded = layer(encoded)

        x = self.pool(encoded)
        for layer in self.ffs:
            x = layer(x)
            
        return self.output_ff(x)

class Transformer(Model):
    def __init__(self):
        self.name = "Transformer"

    def fit(self, train, val=None, neurons=10, epochs=200, lookback=30):
        self.train = train
        self.lookback = lookback
        if val is not None:
            self.val = val

        x, y = sequence(train, lookback)
        if val is not None:
            extended_val = np.append(self.train[-self.lookback:], val)
            x_val, y_val = sequence(extended_val, lookback)

        self.model = Sequential()
        self.model.add(KerasTransformer(
            sequence_length=lookback,
            enc_layers=neurons,
            key_dim=256,
            heads=4,
            ff_dim=256,
            mlp_units=[128, 64],
            dropout=0.05
        ))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=1e-9)
        optimizer = Adam(learning_rate=1e-4)

        self.model.compile(optimizer=optimizer, loss='mse')

        if val is not None:
            self.model.fit(x, y, validation_data=(x_val, y_val), epochs=epochs, callbacks=[reduce_lr])
        else:
            self.model.fit(x, y, epochs=epochs, callbacks=[reduce_lr])

    def predict(self, x):
        if hasattr(self, 'val'):
            extended_data = np.append(self.val[-self.lookback:], x)
        else:
            extended_data = np.append(self.train[-self.lookback:], x)
        inp, _ = sequence(extended_data, self.lookback, 1)
        out = self.model.predict(inp)
        return out