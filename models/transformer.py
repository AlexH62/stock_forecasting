from models.model import Model
from preprocessor import sequence

import numpy as np
import keras
from keras.layers import MultiHeadAttention, Conv1D, Dropout, LayerNormalization
from keras.layers import Dense, GlobalAveragePooling1D

class Transformer(Model):
    def __init__(self):
        self.name = "Transformer"
    
    def fit(self, train, neurons, epochs, lookback=30):
        def transformer_encoder(inputs):
            num_heads = 4
            head_size = 256
            ff_dim = 4

            # Normalization and Attention
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
            x = Dropout(dropout)(x)

            res = x + inputs

            # Feed Forward Part
            x = LayerNormalization(epsilon=1e-6)(res)
            x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
            x = Dropout(dropout)(x)
            x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
            return x + res
        
        mlp_units = [128]
        dropout = 0.25

        inp, out = sequence(train, lookback)
        
        inputs = keras.Input(shape=(inp.shape[1], 1))
        x = inputs
        for _ in range(neurons):
            x = transformer_encoder(x)

        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(dropout)(x)

        # output layer
        outputs = Dense(1)(x)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer="adam", loss='mse')

        self.model.fit(inp, out, epochs=epochs)
    
    def predict(self, x):
        return self.model.predict(x)