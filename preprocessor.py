import numpy as np

def split(data, train_test_split=0.7, validation=True):
    split_idx = int(train_test_split * len(data))

    if validation:
        val_split = (1 - train_test_split) / 2
        val_idx = int(val_split * len(data))
        return data[:split_idx], data[split_idx:split_idx + val_idx], data[split_idx + val_idx:]
    
    return data[:split_idx], data[split_idx:]

def sequence(data, lookback, lookahead=1):
    x, y = [], []

    for i in range(len(data)):
        end_ix = i + lookback
        y_ix = end_ix + lookahead - 1
        
        if y_ix > len(data)-1:
            break
        
        seq_x, seq_y = data[i:end_ix], data[y_ix]
        x.append(seq_x)
        y.append(seq_y)

    x, y = np.array(x), np.array(y)
    x = x.reshape((x.shape[0], x.shape[1], 1))

    return x, y