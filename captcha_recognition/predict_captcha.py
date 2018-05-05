# -*- coding:utf-8 -*-

from PIL import Image
import tensorflow as tf
import numpy as np
import os
from generate_captcha import GenCaptcha
import captcha_model
import argparse

init_op = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
_gen_captcha = GenCaptcha()
width, height, char_num, characters, classes = _gen_captcha.get_parameter()
x = tf.placeholder(tf.float32, [None, height, width, 1])
keep_prob = tf.placeholder(tf.float32)

model = captcha_model.captchaModel(width, height, char_num, classes)
y_conv = model.create_model(x, keep_prob)
predict = tf.argmax(tf.reshape(y_conv, [-1, char_num, classes]), 2)
saver = tf.train.Saver()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chptcha',
                        help='chptcha to regognize')
    parser.add_argument('-p', '--img-path',
                        help='captcha path')
    parser.add_argument('--model', default='capcha_model.ckpt',
                        help='trained model to predict, default: %(default)s')
    args = parser.parse_args()

    if args.chptcha:
        chptchas = [os.path.abspath(args.chptcha)]
    elif args.img_path:
        chptcha_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.img_path)
        chptchas = [os.path.join(chptcha_path, file) for file in os.listdir(chptcha_path)
                    if file.endswith('.jpg') or file.endswith('.png')]
    else:
        raise RuntimeError('no chptcha or path input')
    X_test = []
    for c in chptchas:
        gray_image = Image.open(c).convert('L')
        img = np.array(gray_image.getdata())
        X_test.append(np.reshape(img, [height, width, 1]) / 255.0)

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        saver.restore(sess, args.model)
        pre_list = sess.run(predict, feed_dict={x: X_test, keep_prob: 1})
        for i, pred_y in enumerate(pre_list):
            s = ''.join([characters[char_idx] for char_idx in pred_y])
            print('{} predict: {}'.format(os.path.split(chptchas[i])[-1], s))
