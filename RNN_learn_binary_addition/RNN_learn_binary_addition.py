#!/usr/bin/env python
# -*- coding: utf-8 -*-


__version__ = '0.2.20180328'


import copy
import numpy as np
import sys
sys.path.append('../utils/')
from data_manipulation import train_test_split, shuffle_data


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output * (1 - output)

def gen_binary_seq(binary_dim):
    int2binary = {}
    max_number = pow(2, binary_dim)
    int_list = range(max_number)
    binary = np.unpackbits(
        np.array([int_list], dtype=np.uint8).T, axis=1)
    for i in int_list:
        int2binary[i] = binary[i]
    return int_list, int2binary

def binary2int(binary_seq):
    value = 0
    for i, bit in enumerate(reversed(binary_seq)):
        value += bit * np.power(2, i)
    return value

def gen_data(n_samples, int_list, int2binary, seed=None):
    X, y = [], []
    if seed:
        np.random.seed(seed)
    for i in range(n_samples):
        a = np.random.choice(int_list[:len(int_list)//2])
        b = np.random.choice(int_list[:len(int_list)//2])
        c = a + b
        X.append([int2binary[a], int2binary[b]])
        y.append(int2binary[c])
    return np.array(X), np.array(y)


class RNN(object):

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, seed=None):
        # initialize neural network weights
        if seed:
            np.random.seed(seed)
        self.synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1   # shpae = (2, 16)
        self.synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1  # shpae = (16, 1)
        self.synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1  # shpae = (16, 16)

        self.learning_rate = learning_rate

    def _forward(self, x, y, training=False):
        a = x[0]
        b = x[1]
        c = y
        output = np.zeros_like(a)
        self.overallError = 0
        binary_dim = len(a)

        self.layer_2_deltas = []
        self.layer_1_values = [np.zeros(hidden_dim)]
        for i in range(binary_dim):
            X = np.array([[a[-i - 1], b[-i - 1]]])  # shape = (1, 2)
            layer_1 = sigmoid(np.dot(X, self.synapse_0) + np.dot(self.layer_1_values[-1], self.synapse_h))  # memory state
            layer_2 = sigmoid(np.dot(layer_1, self.synapse_1))  # output, shape = (1, 1)
            output[-i - 1] = np.round(layer_2[0][0])
            self.layer_1_values.append(copy.deepcopy(layer_1))
            if training:
                y = np.array([[c[- i - 1]]]).T
                layer_2_error = y - layer_2
                self.layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
                self.overallError += np.abs(layer_2_error[0])
                # store hidden layer so we can use it in the next timestep

        return output

    def _backprog(self, x):
        a = x[0]
        b = x[1]
        synapse_0_update = np.zeros_like(self.synapse_0)
        synapse_1_update = np.zeros_like(self.synapse_1)
        synapse_h_update = np.zeros_like(self.synapse_h)
        future_layer_1_delta = np.zeros(hidden_dim)
        for i in range(binary_dim):

            X = np.array([[a[i], b[i]]])
            layer_1 = self.layer_1_values[-i - 1]
            prev_layer_1 = self.layer_1_values[-i - 2]
            # error at output layer
            layer_2_delta = self.layer_2_deltas[-i - 1]
            # error at hidden layer
            layer_1_delta = (future_layer_1_delta.dot(self.synapse_h.T) +
                             layer_2_delta.dot(self.synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
            # let's update all our weights so we can try again
            synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
            synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
            synapse_0_update += X.T.dot(layer_1_delta)

            future_layer_1_delta = layer_1_delta

        self.synapse_0 += synapse_0_update * self.learning_rate
        self.synapse_1 += synapse_1_update * self.learning_rate
        self.synapse_h += synapse_h_update * self.learning_rate

    def fit(self, X_train, y_train, batch_size=1, n_epochs=10, display_epoch=10):
        for i in range(1, n_epochs+1):
            X_train, y_train = shuffle_data(X_train, y_train)
            for x, y in zip(X_train, y_train):
                self._forward(x, y, training=True)
                self._backprog(x)
            if i % display_epoch == 0:
                y_pred = self.predict(X_train)
                print('eproch {:<3} training score: {}'.format(i, self.accuracy(y_train, y_pred)))
        print('-------------------------------')
        print('Training Finished!')

    def accuracy(self, y, y_pred):
        return np.mean([np.all(np.asarray(_y) == np.asarray(_y_pred))
                        for _y, _y_pred in zip(y, y_pred)])

    def predict(self, X_test):
        return np.array([self._forward(x, [], training=False) for x in X_test])

    def summary(self):
        pass

if __name__ == '__main__':

    binary_dim = 8
    int_list, int2binary = gen_binary_seq(binary_dim)

    # input variables
    learning_rate = 0.1
    input_dim = 2
    hidden_dim = 16
    output_dim = 1

    n_samples = 1000
    seed = 0
    X, y = gen_data(n_samples, int_list, int2binary, seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, seed=seed)

    clf = RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
              learning_rate=learning_rate, seed=seed)
    clf.fit(X_train, y_train, n_epochs=20, batch_size=1, display_epoch=2)

    y_test_pred = clf.predict(X_test)

    print('\nTest accuray: {:.4g} %'.format(clf.accuracy(y_test, y_test_pred) * 100))
    print('-------------------------------')
    for i in range(5):
        print('True binary:', str(y_test[i]))
        print('Pred binary:', str(y_test_pred[i]))
        a, b, c, d = map(binary2int, [X_test[i][0], X_test[i][1], y_test[i], y_test_pred[i]])
        print('guess {} + {} = {}, right answer {}'.format(a, b, d, c))
        print('-------------------------------')

'''
eproch 2   training score: 0.00857142857143
eproch 4   training score: 0.01
eproch 6   training score: 0.355714285714
eproch 8   training score: 0.764285714286
eproch 10  training score: 0.834285714286
eproch 12  training score: 1.0
eproch 14  training score: 1.0
eproch 16  training score: 1.0
eproch 18  training score: 1.0
eproch 20  training score: 1.0
-------------------------------
Training Finished!

Test accuray: 100 %
-------------------------------
('True binary:', '[1 0 0 0 0 1 0 0]')
('Pred binary:', '[1 0 0 0 0 1 0 0]')
guess 125 + 7 = 132, right answer 132
-------------------------------
('True binary:', '[0 1 1 0 1 0 0 1]')
('Pred binary:', '[0 1 1 0 1 0 0 1]')
guess 31 + 74 = 105, right answer 105
-------------------------------
('True binary:', '[1 0 0 0 1 0 0 0]')
('Pred binary:', '[1 0 0 0 1 0 0 0]')
guess 42 + 94 = 136, right answer 136
-------------------------------
('True binary:', '[0 1 1 0 0 1 0 1]')
('Pred binary:', '[0 1 1 0 0 1 0 1]')
guess 46 + 55 = 101, right answer 101
-------------------------------
('True binary:', '[1 0 1 0 0 1 1 0]')
('Pred binary:', '[1 0 1 0 0 1 1 0]')
guess 100 + 66 = 166, right answer 166
-------------------------------
'''
