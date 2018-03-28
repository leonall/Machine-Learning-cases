#!/usr/bin/env python
# -*- coding: utf-8 -*-


__version__ = '0.1.20180328'

import numpy as np
import sys
import tensorflow as tf
from tensorflow.contrib import rnn as rnn_cell
sys.path.append('../utils/')
from data_manipulation import train_test_split, shuffle_data
from RNN_learn_binary_addition import gen_binary_seq, gen_data, binary2int


class HParam(object):
    batch_size = 64
    seq_length = 8
    num_layers = 2
    state_size = 16
    learning_rate = 0.01

class RNN(object):
    def __init__(self, sess, args, seed=None):
        # initialize neural network weights
        if seed:
            tf.random.seed(seed)
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.num_layers = args.num_layers
        self.state_size = args.state_size
        self.learning_rate = args.learning_rate
        self.sess = sess

        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, [self.batch_size, self.seq_length, 2])
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.seq_length])

        with tf.name_scope('model'):
            self.cell = rnn_cell.BasicRNNCell(num_units=self.state_size)
            def _get_cell(state_size):
                return rnn_cell.BasicRNNCell(num_units=state_size)
            self.cell = rnn_cell.MultiRNNCell([_get_cell(self.state_size) for _ in range(self.num_layers)])
            self.initial_state = self.cell.zero_state(
                self.batch_size, tf.float32)
            outputs, last_state = tf.nn.dynamic_rnn(
                self.cell, self.X, initial_state=self.initial_state)

        with tf.name_scope('loss'):
            weights = tf.Variable(tf.truncated_normal([self.state_size, 1], stddev=0.01))
            bias = tf.zeros([1])
            outputs = tf.reshape(outputs, [-1, self.state_size])
            logits = tf.sigmoid(tf.matmul(outputs, weights) + bias)
            self.predictions = tf.reshape(logits, [-1, binary_dim])
            self.y_pred = tf.round(self.predictions)
            self.cost = tf.losses.mean_squared_error(self.y, self.predictions)
            # targets = tf.reshape(self.y, [-1])
            tf.summary.scalar('loss', self.cost)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.cast(tf.reduce_sum(self.y_pred, axis=1), tf.float32),
                                          tf.cast(tf.reduce_sum(self.y, axis=1), tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def predict(self, x):
        return self.sess.run(self.y_pred, feed_dict={self.X: x})

    def fit(self, X_train, y_train, n_epochs=10, display_epoch=10):

        self.sess.run(tf.global_variables_initializer())
        for i in range(1, n_epochs+1):
            X_train, y_train = shuffle_data(X_train, y_train)
            for j in range(0, X_train.shape[0], self.batch_size)[:-1]:
                x = X_train[j: j + self.batch_size]
                y = y_train[j: j + self.batch_size]
                loss, acc, _ = self.sess.run([self.cost, self.accuracy, self.optimizer],
                                             feed_dict={self.X: x, self.y: y})
            if i % display_epoch == 0:
                print('eproch {:<3}, training score: {}'.format(i, self.get_accuracy(X_train, y_train)))
        print('-------------------------------')
        print('Training Finished!')

    def get_accuracy(self, X_test, y_test):
        Acc = []
        for j in range(0, X_test.shape[0], self.batch_size)[:-1]:
            x = X_test[j: j + self.batch_size]
            y = y_test[j: j + self.batch_size]
            acc = self.sess.run(self.accuracy, feed_dict={self.X: x, self.y: y})
            Acc.append(acc)
        return sess.run(tf.reduce_mean(Acc))

if __name__ == '__main__':

    binary_dim = 8
    int_list, int2binary = gen_binary_seq(binary_dim)
    n_samples = 1000
    seed = 0
    args = HParam()
    X, y = gen_data(n_samples, int_list, int2binary, seed)

    X = np.flip(X, axis=-1)  # flip
    y = np.flip(y, axis=-1)
    X = X.transpose((0, 2, 1))  # [n, 2, 8] ==> [n, 8, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, seed=seed)
    print(X_train.shape, y_train.shape)

    with tf.Session() as sess:
        clf = RNN(sess=sess, args=args, seed=seed)
        clf.fit(X_train, y_train, n_epochs=20, display_epoch=2)
        print('-------------------------------')
        print('\nTest accuray: {:.4g} %'.format(clf.get_accuracy(X_test, y_test) * 100))
        print('-------------------------------')

        y_pred = clf.predict(X_test[0: clf.batch_size])
        y_pred = sess.run(clf.y_pred, feed_dict={clf.X: X_test[0: clf.batch_size]})
        for i in range(5):
            print(X_test[i].shape)
            x_1_binary = np.flip(X_test[i][:, 0], axis=0)
            x_2_binary = np.flip(X_test[i][:, 1], axis=0)
            y_sample_binary = np.flip(y_test[i], axis=0)
            y_pred_binary = np.flip(y_pred[i], axis=0).astype(np.int32)
            print('True binary:', str(y_sample_binary))
            print('Pred binary:', str(y_pred_binary))
            a, b, c, d = map(binary2int, [x_1_binary, x_2_binary, y_pred_binary, y_sample_binary])
            print('guess {} + {} ==> {}, right answer == {}'.format(a, b, c, d))
            print('-------------------------------')

'''
eproch 2  , training score: 0.19218750298023224
eproch 4  , training score: 0.17812499403953552
eproch 6  , training score: 0.30781251192092896
eproch 8  , training score: 0.9937499761581421
eproch 10 , training score: 1.0
eproch 12 , training score: 1.0
eproch 14 , training score: 1.0
eproch 16 , training score: 1.0
eproch 18 , training score: 1.0
eproch 20 , training score: 1.0
-------------------------------
Training Finished!
-------------------------------

Test accuray: 100 %
-------------------------------
(8, 2)
True binary: [1 1 0 1 1 0 0 0]
Pred binary: [1 1 0 1 1 0 0 0]
guess 123 + 93 ==> 216, right answer == 216
-------------------------------
(8, 2)
True binary: [0 1 1 0 1 0 1 1]
Pred binary: [0 1 1 0 1 0 1 1]
guess 63 + 44 ==> 107, right answer == 107
-------------------------------
(8, 2)
True binary: [0 0 1 0 0 1 0 0]
Pred binary: [0 0 1 0 0 1 0 0]
guess 34 + 2 ==> 36, right answer == 36
-------------------------------
(8, 2)
True binary: [0 1 0 0 0 1 0 0]
Pred binary: [0 1 0 0 0 1 0 0]
guess 33 + 35 ==> 68, right answer == 68
-------------------------------
(8, 2)
True binary: [0 0 1 0 0 1 1 1]
Pred binary: [0 0 1 0 0 1 1 1]
guess 16 + 23 ==> 39, right answer == 39
-------------------------------
'''
