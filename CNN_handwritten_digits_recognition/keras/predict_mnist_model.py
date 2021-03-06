#!/usr/bin/python
# -*- coding: utf-8 -*

from __future__ import division
from __future__ import print_function
import numpy as np
from keras.models import load_model
from PIL import Image, ImageFilter
import os
import argparse

def load_data(digit_image):
    grayimage = Image.open(digit_image).convert('L')
    width = float(grayimage.size[0])
    height = float(grayimage.size[1])
    if width != 28 and height != 28:
        print('{} need resize into 28 x 28'.format(os.path.split(digit_image)[-1]))
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
        out.save(digit_image[:-4]+ '_28pix' + '.png')
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    else:
        tva = [(255 - x) * 1.0 / 255.0 for x in list(grayimage.getdata())]
    return np.array(tva)


def main(argv, trained_model):
    model = load_model(trained_model)
    if os.path.isfile(argv):
        imvalue = load_data(argv)
        y_predict = model.predict(np.array([imvalue.reshape(28, 28, 1)], dtype=np.float32), batch_size=1)
        predint = np.argmax(y_predict, axis=1)
        print('\nguess the digit is {}\n'.format(predint[0]))
    else:
        if os.path.isdir(argv):
            images_path = argv
        else:
            images_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hand_written_digits')
        images_name = []
        images_file = []
        for image in os.listdir(images_path):
            if image.endswith('png') and '28pix' not in image:
                images_name.append(image)
                image_file = os.path.join(images_path, image)
                images_file.append(load_data(image_file).reshape(28, 28, 1))
        images_file = np.array(images_file, dtype=np.float32)
        y_predict = model.predict(images_file)
        pred_digits = np.argmax(y_predict, axis=1)
        for image, pred_digit in zip(images_name, pred_digits):
            print(image, pred_digit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', default='mnist_keras.h5',
                        help="keras trained model")
    parser.add_argument('--image', default='',
                        help='digit image or images path to predict')
    args = parser.parse_args()

    main(args.image, args.model)
