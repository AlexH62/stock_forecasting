from models.model import Model
from preprocessor import sequence

import keras
import numpy as np
from keras import layers

class Transformer(Model):
    def __init__(self):
        self.name = "Transformer"

    def __transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
    
    def __build_model(
            self,
            input_shape,
            head_size,
            num_heads,
            ff_dim,
            num_transformer_blocks,
            mlp_units,
            dropout=0,
            mlp_dropout=0,
        ):
            inputs = keras.Input(shape=input_shape)
            x = inputs
            for _ in range(num_transformer_blocks):
                x = self.__transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

            x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
            for dim in mlp_units:
                x = layers.Dense(dim, activation="relu")(x)
                x = layers.Dropout(mlp_dropout)(x)
            outputs = layers.Dense(1, activation="relu")(x)
            return keras.Model(inputs, outputs)

    def fit(self, train, neurons, epochs, lookback=30):
        self.train = train
        self.lookback = lookback

        x, y = sequence(train, lookback)

        self.model = self.__build_model(
            input_shape = x.shape[1:],
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=neurons,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )

        self.model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=1e-4)
        )

        #callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=64,
            #callbacks=callbacks,
        )

    def predict(self, x):
        extended_data = np.append(self.train[-self.lookback:], x)
        inp, _ = sequence(extended_data, self.lookback, 1)
        return self.model.predict(inp)