#!/usr/bin/python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import tempfile
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import mnist_model
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mnist_data'))
from maybe_download import maybe_download

tf_config = tf.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 分配50%
tf_config.gpu_options.allow_growth = True  # 自适应
session = tf.Session(config=tf_config)

maybe_download()

def main(batch_size, epochs):
    mnist = input_data.read_data_sets(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', 'mnist_data'), one_hot=True)

    # 输入变量，mnist图片大小为28*28
    x = tf.placeholder(tf.float32, [None, 784])
    # 输出变量，数字是1-10
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 构建网络，输入—>第一层卷积—>第一层池化—>第二层卷积—>第二层池化—>第一层全连接—>第二层全连接
    y_conv, keep_prob = mnist_model.deepnn(x)

    # 第一步对网络最后一层的输出做一个softmax，第二步将softmax输出和实际样本做一个交叉熵
    # cross_entropy返回的是向量
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)

    # 求cross_entropy向量的平均值得到交叉熵
    cross_entropy = tf.reduce_mean(cross_entropy)

    # AdamOptimizer是Adam优化算法：一个寻找全局最优点的优化算法，引入二次方梯度校验
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 在测试集上的精确度
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # 将神经网络图模型保存本地，可以通过浏览器查看可视化网络结构
    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    # 将训练的网络保存下来
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1, epochs * mnist.train.num_examples // batch_size + 1):
            batch = mnist.train.next_batch(batch_size)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})  # 输入是字典，表示tensorflow被feed的值
                print('step %d, training accuracy %g' % (i, train_accuracy))
            if i % (mnist.train.num_examples // batch_size) == 0:
                print('###### epoch {} finished! ######'.format((i * batch_size) // mnist.train.num_examples + 1))
                test_accuracy = 0
                batches = mnist.test.num_examples // batch_size
                for i in range(batches):
                    batch = mnist.test.next_batch(batch_size)
                    test_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) / batches
                print('testing accuracy %g' % (train_accuracy))
                print('########################')
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        test_accuracy = 0
        batches = mnist.test.num_examples // batch_size
        for i in range(batches):
            batch = mnist.test.next_batch(batch_size)
            test_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) / batches

        print('test accuracy %g' % test_accuracy)

        saver.save(sess, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_cnn_model_{}.ckpt".format(epochs)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int,
                        help='batch-size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='batch-size')
    args = parser.parse_args()
    main(args.batch_size, args.epochs)
