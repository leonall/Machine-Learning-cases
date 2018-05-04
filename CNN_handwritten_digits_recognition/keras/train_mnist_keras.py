#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import load_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mnist_data'))
from maybe_download import maybe_download

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--epochs', default=20, type=int,
                    help="epochs")
parser.add_argument('--batch-size', default=86, type=int,
                    help='batch-size')
args = parser.parse_args()

epochs = args.epochs  # Turn epochs to 30 to get 0.995 accuracy
batch_size = args.batch_size

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)
sns.set(style='white', context='notebook', palette='deep')

# Load the data
maybe_download()
mnist = input_data.read_data_sets('../mnist_data', one_hot=True)
X_train, X_val, Y_train, Y_val = mnist.train.images, mnist.validation.images, mnist.train.labels, mnist.validation.labels
X_test, Y_test = mnist.test.images, mnist.test.labels

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(X_val, Y_val),
                              verbose=2, steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction])
model.save('mnist_keras_{}.h5'.format(epochs))

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.savefig('history_figure.png')
plt.close()


def plot_confusion_matrix(cm, classes, figure_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(figure_name)
    plt.close()

# Predict the values from the validation dataset
Y_pred_val = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_val = np.argmax(Y_pred_val, axis=1)
# Convert validation observations to one hot vectors
Y_true_val = np.argmax(Y_val, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true_val, Y_pred_val)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(10), figure_name='val_set_confusion_matrix.png')


Y_pred_test = model.predict(X_test)
Y_pred_test = np.argmax(Y_pred_test, axis=1)
Y_true_test = np.argmax(Y_test, axis=1)
confusion_mtx = confusion_matrix(Y_true_test, Y_pred_test)
plot_confusion_matrix(confusion_mtx, classes=range(10), figure_name='test_set_confusion_matrix.png')
print('test acc: {}'.format(accuracy_score(Y_true_test, Y_pred_test)))
