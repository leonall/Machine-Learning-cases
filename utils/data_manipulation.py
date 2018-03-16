#!/usr/bin/env python
# -*- coding: utf-8 -*-


__version__ = '0.1.20180315'


import numpy as np
import math

def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    idx = np.arange(X.shape[0])
    if seed:
        np.random.seed(seed)
    np.random.shuffle(idx)
    return X[idx], y[idx]


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    split_id = int(math.ceil(X.shape[0] * (1 - test_size)))
    X_train, X_test = X[:split_id], X[split_id:]
    y_train, y_test = y[:split_id], y[split_id:]
    return X_train, X_test, y_train, y_test
