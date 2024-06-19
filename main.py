import repository
import utils
import metrics

import argparse
import numpy as np
from models.gru import GRU
from models.lstm import LSTM
from models.transformer import Transformer
from models.node import NODE
from models.anode import ANODE
from models.odetransformer import ODETransformer
from sklearn.preprocessing import RobustScaler

TICKERS = ['NVDA', 'IBM', 'AAPL', 'NFLX', 'GOOG', 'GS', 'JPM', 'BCS', 'SAN', 'MS']
START = '2021-01-01'
END = '2023-01-01'

parser = argparse.ArgumentParser("Model training")
parser.add_argument("model_type", help="Model type to use. Options are lstm, gru, transformer, node, anode, odetransformer", type=str)
parser.add_argument("depth", help="Number of repeated modules to use", type=int)
parser.add_argument("epochs", help="Maximum number of epochs", type=int)
parser.add_argument("horizon", help="Forecasting horizon", type=int)
args = parser.parse_args()

HORIZON = args.horizon

rmses = []
maes = []
mapes = []
r2s = []
for ticker in TICKERS:
    data = repository.get_data(ticker, START, END)
    scaler = RobustScaler()

    train, val, test = utils.split(data)

    scaled_train = scaler.fit_transform(train.reshape(-1, 1))
    scaled_val = scaler.transform(val.reshape(-1, 1))
    scaled_test = scaler.transform(test.reshape(-1, 1))

    match args.model_type:
        case "lstm":
            model = LSTM()
        case "gru":
            model = GRU()
        case "transformer":
            model = Transformer()
        case "node":
            model = NODE()
        case "anode":
            model = ANODE()
        case "odetransformer":
            model = ODETransformer()

    model.fit(scaled_train, val=scaled_val, depth=args.depth, lookback=60, epochs=args.epochs, horizon=HORIZON)

    predictions = model.predict(scaled_test)

    unscaled_prediction = scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()

    print(f'Metrics for {ticker}')
    rmse, mae, mape, r2 = metrics.print_all(test, unscaled_prediction)

    rmses.append(rmse)
    maes.append(mae)
    mapes.append(mape)
    r2s.append(r2)

    train_val = np.append(train, val)
    utils.plot(train_val, test, unscaled_prediction, model.name, ticker, HORIZON)

utils.write_to_csv('results.csv', model.name, TICKERS, maes, mapes, rmses, r2s)