from abc import ABC, abstractmethod

# Abstract base class for our models to implement, used to make training and testing smoother
class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, train, val=None, depth=None, epochs=None):
        pass

    @abstractmethod
    def predict(self, x):
        pass