import numpy as np
import singularity as S
from singularity.components import layers, optimizers, regularizers, models
from singularity.utils import datasets
from time import sleep

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

model = models.OverlayModel()

model.add(layers.InputLayer((None, 1, img_rows, img_cols)))
model.add(layers.Conv2DLayer(nb_filters,(nb_conv, nb_conv), padding = 0, activation = S.relu))
model.add(layers.Conv2DLayer(nb_filters,(nb_conv, nb_conv), padding = 0,max_pool_shape = (nb_pool, nb_pool), activation = S.relu))
model.add(layers.DropoutLayer(0.25))
model.add(layers.FlattenLayer())
#model.add(layers.Dense(128,activation = S.relu))
model.add(layers.DenseLayer(144,activation = S.relu))
model.add(layers.DropoutLayer(0.5))
model.add(layers.DenseLayer(nb_classes, activation = S.softmax))
    
adam = optimizers.Adam(learning_rate = 0.02)
adadelta = optimizers.AdaDELTA()
adagrad = optimizers.Adagrad(learning_rate = 0.002)
sgd = optimizers.SGD(learning_rate = 0.002)
rmsprop = optimizers.RMSProp(learning_rate = 0.001)
#model.build(loss=categorical_cross_entropy, optimizer = rmsprop, init_params = glorot_uniform, init_bias = zeros, regularizers = [L1(0.001),L2(0.0001)])
model.build(loss=S.categorical_cross_entropy, optimizer = adam, regularizers = [regularizers.L2()]) # L2 weight decay

model.train(X_train,Y_train, batch_size= batch_size, iterations = 1)
accuracy = model.accuracy(X_test,np.argmax(Y_test, axis=1))
print (accuracy)
