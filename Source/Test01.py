import numpy as np
import singularity as S
from singularity.components import layers, optimizers, regularizers, models
from singularity.utils import datasets

batch_size = 128
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = datasets.load_mnist()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype(S.floatX())
X_test = X_test.astype(S.floatX())
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = S.categorical(y_train, 10)
Y_test = S.categorical(y_test, 10)

model = models.DeepNetwork()

model.add(layers.InputLayer((None,28*28)))
model.add(layers.DenseLayer(512))
model.add(layers.ActivationLayer(S.relu))
model.add(layers.DenseLayer(484, activation = S.relu))
model.add(layers.DropoutLayer(0.2))
model.add(layers.DenseLayer(512, activation = S.relu))
model.add(layers.DropoutLayer(0.2))
model.add(layers.DenseLayer(10, activation = S.softmax))

adam = optimizers.Adam(learning_rate = 0.02)
adadelta = optimizers.AdaDELTA()
adagrad = optimizers.Adagrad(learning_rate = 0.002)
sgd = optimizers.SGD(learning_rate = 0.002)
rmsprop = optimizers.RMSProp(learning_rate = 0.001)
#model.build(loss=categorical_cross_entropy, optimizer = rmsprop, init_params = glorot_uniform, init_bias = zeros, regularizers = [L1(0.001),L2(0.0001)])
model.build(loss=S.categorical_cross_entropy, optimizer = rmsprop, init_weights = S.glorot_uniform, init_bias = S.zeros, regularizers = [regularizers.L2()])

model.train(X_train,Y_train, batch_size= batch_size, iterations = 1)
accuracy = model.accuracy(X_test,np.argmax(Y_test, axis=1))
print (accuracy)
 