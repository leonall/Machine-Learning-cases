# -*- coding:utf-8 -*-

from generate_poetry import Poetry
from poetry_model import poetryModel
import tensorflow as tf
import os
import time


if __name__ == '__main__':
    t0 = time.time()
    batch_size = 256
    epoch = 20
    rnn_size = 128
    num_layers = 2
    poetrys = Poetry()
    words_size = len(poetrys.word_to_id)
    inputs = tf.placeholder(tf.int32, [batch_size, None])
    targets = tf.placeholder(tf.int32, [batch_size, None])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    model = poetryModel()
    logits, probs, initial_state, last_state = model.create_model(inputs, batch_size,
                                                                  rnn_size, words_size, num_layers, True, keep_prob)
    loss = model.loss_model(words_size, targets, logits)
    learning_rate = tf.Variable(0.0, trainable=False)
    optimizer = model.optimizer_model(loss, learning_rate)
    saver = tf.train.Saver()
    loss_record = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'loss.tab')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(learning_rate, 0.002 * 0.97))
        next_state = sess.run(initial_state)
        step = 1
        t1 = time.time()
        while True:
            x_batch, y_batch = poetrys.next_batch(batch_size)
            feed = {inputs: x_batch, targets: y_batch, initial_state: next_state, keep_prob: 0.5}
            train_loss, _, next_state = sess.run([loss, optimizer, last_state], feed_dict=feed)
            with open(loss_record, 'a') as fout:
                fout.write('{}\t{:.4g}\n'.format(step, train_loss))
            if step % 100 == 0:
                print("step:{} loss: {:.4g} in {:.4g} sec.".format(step, train_loss, time.time() - t1))
                t1 = time.time()
            if step > 200:
                break
            if step % 1000 == 0:
                n = step/1000
                sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** n)))
            step += 1
        saver.save(sess, os.path.join(os.path.dirname(os.path.abspath(__file__)), "poetry_model.ckpt"))
    print('Finish training in {:.4g} sec'.format(time.time() - t0))
