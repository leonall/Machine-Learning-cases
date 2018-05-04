import tensorflow as tf
import urllib
import os


WORK_DIRECTORY = '.'
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def maybe_download(filename='.'):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    else:
        print('Successfully finded the mnist data')
    return


if __name__ == '__main__':
    maybe_download()
