from models.model import Model
from preprocessor import sequence

import numpy as np
from tfdiffeq.models import ODENet

class NODE(Model):
    def __init__(self):
        self.name = "NODE"

    def fit(self, train, val=None, neurons=10, epochs=200, lookback=30):
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
            hidden_dim=1024, 
            output_dim=1,
            augment_dim=0,
            adjoint=False,
            solver="dopri5"
        )

        self.model.compile(optimizer="adam", loss='mse')

        if val is not None:
            self.model.fit(x, y, validation_data=(x_val, y_val), epochs=epochs)
        else:
            self.model.fit(x, y, epochs=epochs)

    def predict(self, data):
        if self.val is not None:
            extended_data = np.append(self.val[-self.lookback:], data)
        else:
            extended_data = np.append(self.train[-self.lookback:], data)
        inp, _ = sequence(extended_data, self.lookback, 1)
        inp = np.expand_dims(inp.squeeze(), axis=1)
        
        return self.model(inp, training=False).numpy()