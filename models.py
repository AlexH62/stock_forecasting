from ncps import wirings
from ncps.tf import LTC, CfC
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM, GRU, MultiHeadAttention, Conv1D, Dropout, LayerNormalization
from keras.layers import Dense, GlobalAveragePooling1D
import keras

class Models:
  def __init__(self):
    pass

  def LSTM(self, units, steps, n_features):
    model = Sequential()
    model.add(LSTM(units, input_shape=(steps, n_features)))
    model.add(Dense(1))
    #optim = Adam(learning_rate=0.0001)
    model.compile(optimizer="adam", loss='mse')
    return "LSTM", model
  
  def GRU(self, units, steps, n_features):
    model = Sequential()
    model.add(GRU(units, input_shape=(steps, n_features)))
    model.add(Dense(1))
    #optim = Adam(learning_rate=0.0001)
    model.compile(optimizer="adam", loss='mse')
    return "GRU", model
  
  def Transformer(self, units, steps, n_features, mlp_units=[128], dropout=0.25):
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
    
    inputs = keras.Input(shape=(steps, n_features))
    x = inputs
    for _ in range(units):
        x = transformer_encoder(x)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(dropout)(x)

    # output layer
    outputs = Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss='mse')

    return "Transformer", model
  
  def LTC(self, units, steps, n_features):
    wiring = wirings.AutoNCP(units ,1)
    model = keras.models.Sequential(
      [
        keras.layers.InputLayer(input_shape=(None, n_features)),
        CfC(wiring),
      ]
    )
    model.compile(optimizer="adam", loss='mse')

    return "LTC", model