#!/usr/bin/python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import mnist_model
from PIL import Image, ImageFilter
import argparse


def load_data(argv):
    grayimage = Image.open(argv).convert('L')
    width = float(grayimage.size[0])
    height = float(grayimage.size[1])
    if width != 28 and height != 28:
        print(width, height)
        newImage = Image.new('L', (28, 28), (255))

        if width > height:
            nheight = int(round((20.0 / width * height), 0))
            if (nheight == 0):
                nheight = 1
            img = grayimage.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - nheight) / 2), 0))
            newImage.paste(img, (4, wtop))
        else:
            nwidth = int(round((20.0 / height * width), 0))
            if (nwidth == 0):
                nwidth = 1
            img = grayimage.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - nwidth) / 2), 0))
            newImage.paste(img, (wleft, 4))

        tv = np.array(list(newImage.getdata()), dtype=np.float32)
        out = Image.fromarray(np.uint8(tv.reshape(28, 28)))
        out.save(argv[:-4]+ '_28pix' + '.png')
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    else:
        tva = [(255 - x) * 1.0 / 255.0 for x in list(grayimage.getdata())]
    return tva


def main(argv, model):
    if os.path.isfile(argv):
        images_file = [load_data(argv)]
    elif os.path.isdir(argv):
        images_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hand_written_digits')
        images_name = []
        images_file = []
        for image in os.listdir(images_path):
            if image.endswith('png') and '28pix' not in image:
                images_name.append(image)
                image_file = os.path.join(images_path, image)
                images_file.append(load_data(image_file))
    images_file = np.array(images_file, dtype=np.float32)
    x = tf.placeholder(tf.float32, [None, 784])
    y_conv, keep_prob = mnist_model.deepnn(x)
    y_predict = tf.nn.softmax(y_conv)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, os.path.join(os.path.dirname(os.path.abspath(__file__)), model))
        prediction = tf.argmax(y_predict, axis=1)
        pred_digits = sess.run(prediction, feed_dict={x: images_file, keep_prob: 1.0})
        print(pred_digits.shape)
        for image, pred_digit in zip(images_name, pred_digits):
            print(image, pred_digit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', default='mnist_cnn_model.ckpt',
                        help="keras trained model")
    parser.add_argument('--image',
                        help='digit image to predict')
    parser.add_argument('--image-path', default='../hand_written_digits',
                        help='images path to predict')
    args = parser.parse_args()

    main(args.image or args.image_path, model=args.model)
