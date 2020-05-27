import numpy as np


def load_data(path):
    X_train = np.load(f'{path}/X_train.npy')
    y_train = np.load(f'{path}/y_train.npy')
    X_test = np.load(f'{path}/X_test.npy')
    y_test = np.load(f'{path}/y_test.npy')

    return X_train, y_train, X_test, y_test
