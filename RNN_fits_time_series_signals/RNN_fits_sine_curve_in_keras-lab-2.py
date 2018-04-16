#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../utils')
from data_manipulation import train_test_split, shuffle_data
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.metrics import accuracy_score


__version__ = '0.1.20180416'


def gen_data(seq_length, n_samples, amp=1):
    X = []
    y = []
    for i in range(n_samples):
        X.append((amp * np.sin(np.linspace(i, seq_length+i, seq_length) / seq_length)).reshape(-1, 1))
        y.append([amp * np.sin((seq_length + i + 1) / seq_length)])
    return np.array(X), np.array(y)


seq_length = 10
timesteps = seq_length
data_dim = 1
n_samples = 5000
seed=0
X, y = gen_data(seq_length=seq_length, n_samples=n_samples)  # X [n_samples, seq_length, data_dim]
                                                             # y [n_samples, 1, data_dim]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, seed=seed)

model = Sequential()
model.add(LSTM(50, input_shape=(timesteps, data_dim), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(10, return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=512, verbose=0)

y_pred = model.predict(X_test, batch_size=100)

test_loss = np.sqrt(np.mean((y_test.reshape(-1) - y_pred.reshape(-1))**2))
print('test lost {}'.format(test_loss))

nums = 100
x0 = np.sin(np.linspace(0, nums-1, nums) / seq_length).reshape(-1, 1)
x = np.sin(np.linspace(0, seq_length - 1, seq_length) / seq_length).reshape(-1, 1)
while x.shape[0] < nums:
    next_x = model.predict(np.array([x[-seq_length:]]), batch_size=1)
    x = np.vstack([x, next_x])
plt.plot(x.reshape(-1), label='learning')
plt.plot(x0.reshape(-1), 'ro', label='original', ms=4, alpha=0.5)
plt.text(0.5, 0.5, 'testing loss :{:.4g}'.format(test_loss),
         ha='left', va='center', transform=plt.gca().transAxes, fontsize=12)
plt.legend()
plt.grid()
plt.savefig(os.path.join(os.path.dirname(__file__), 'images', os.path.splitext(__file__)[0] + '.png'))
plt.show()
