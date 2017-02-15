import numpy as np
import matplotlib.pyplot as plt
import singularity as S
from singularity.components.layers import *
from singularity.components.optimizers import *
from singularity.components.regularizers import *
from singularity.components.models import *
from singularity.utils import datasets
import h5py

from scipy.misc import imsave
import time
import os

batch_size = 512
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = datasets.load_mnist()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype(S.floatX())
X_test = X_test.astype(S.floatX())
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = S.categorical(y_train, nb_classes)
Y_test = S.categorical(y_test, nb_classes)

train = True


# build the VGG16 network
model = OverlayModel()
model.add(InputLayer((None, 1, img_rows, img_cols)))
model.add(PaddingLayer(1))

model.add(Conv2DLayer(32, (3, 3), activation=S.relu, name='conv1_1'))
model.add(PaddingLayer(1))
model.add(Conv2DLayer(8, (3, 3), activation=S.relu, max_pool_shape = (2, 2), name='conv1_2'))

model.add(PaddingLayer(1))
model.add(Conv2DLayer(4, (3, 3), activation=S.relu, name='conv2_1'))
model.add(PaddingLayer(1))
model.add(Conv2DLayer(4, (3, 3), activation=S.relu, max_pool_shape = (2, 2), name='conv2_2'))
 
if (train):
    model.add(FlattenLayer())
    model.add(DenseLayer(500, activation=S.relu))
    model.add(DropoutLayer(0.5))
    model.add(DenseLayer(10, activation=S.softmax))

adam = Adam(learning_rate = 0.02)
model.build(loss=S.categorical_cross_entropy, optimizer = adam, regularizers = [L2()]) # L2 weight decay
       
#model.train(X_train[0:],Y_train, batch_size= batch_size, iterations = 1)
model.train(X_train[0:batch_size * 3],Y_train[0:batch_size * 3], batch_size= batch_size, iterations = 1)
accuracy = model.accuracy(X_test,np.argmax(Y_test, axis=1))
print (accuracy)