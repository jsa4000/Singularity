from .file import *
import urllib
import os.path

##################
#  MNIST Dataset #
##################

def load_mnist(file_path = "./data/mnist.pkl.gz"):
    if (not os.path.exists(file_path)):
        urllib.URLopener().retrieve("https://s3.amazonaws.com/img-datasets/mnist.pkl.gz", file_path)
    return load(file_path) # (X_train, y_train), (X_test, y_test)

