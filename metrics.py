import numpy as np

def print_RMSE(actual, predicted):
    rmse = np.sqrt(np.sum(np.square(actual - predicted)) / len(actual))
    print(f'RMSE: {rmse}')
    return rmse

def print_MAE(actual, predicted):
    mae = np.sum(np.abs(actual - predicted)) / len(actual)
    print(f'MAE: {mae}')
    return mae

def print_MAPE(actual, predicted):
    mape = np.sum(np.abs(np.divide(actual - predicted, actual))) / len(actual)
    print(f'MAPE: {mape}')
    return mape

def print_R2(actual, predicted):
    num = np.sum(np.square(actual - predicted))
    denom = np.sum(np.square(actual - np.mean(actual)))
    r2 = 1 - (num / denom)
    print(f'R Squared: {r2}')
    return r2

def print_all(actual, predicted):
    rmse = print_RMSE(actual, predicted)
    mae = print_MAE(actual, predicted)
    mape = print_MAPE(actual, predicted)
    r2 = print_R2(actual, predicted)

    return rmse, mae, mape, r2