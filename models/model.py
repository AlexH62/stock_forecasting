from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, train, neurons, epochs):
        pass

    @abstractmethod
    def predict(self, x):
        pass