from utils import sequence

import numpy as np
from abc import ABC, abstractmethod

# Abstract base class for our models to implement, used to make training and testing smoother
class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, train, val=None, depth=None, epochs=None, horizon=None):
        pass

    def predict(self, x):
        if hasattr(self, 'val'):
            pre_data = np.append(self.train, self.val)
        else:
            pre_data = self.train
        extended_data = np.append(pre_data[-self.lookback-self.horizon+1:], x)
        inp, _ = sequence(extended_data, self.lookback, self.horizon)
        out = self.model.predict(inp)
        return out