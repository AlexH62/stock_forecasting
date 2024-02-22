from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Preprocessor:
  def __init__(self):
    self.scaler = MinMaxScaler(feature_range=(0, 1))

  def reverse_transform(self, data):
    return self.scaler.inverse_transform(data.reshape(-1, 1))
  
  def scale(self, data, train_test_split=0.8, return_split=False):
    assert train_test_split <= 1
    assert train_test_split > 0

    train_samples = int(train_test_split * data.shape[0])
    train = self.scaler.fit_transform(data[:train_samples].reshape(-1, 1)).squeeze()
    test = self.scaler.transform(data[train_samples:].reshape(-1, 1)).squeeze()

    if return_split:
      return train, test
    return np.append(train, test)

  def sequence(self, sequence, n_steps, train_test_split=0.8):
    X, y = [], []

    for i in range(len(sequence)):
      end_ix = i + n_steps
      
      if end_ix > len(sequence)-1:
        break
      
      seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
      X.append(seq_x)
      y.append(seq_y)

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    train_samples = int(train_test_split * X.shape[0])
    X_train = X[:train_samples]
    y_train = y[:train_samples]

    X_test = X[train_samples:]
    y_test = y[train_samples:]

    return X_train, y_train, X_test, y_test

  



