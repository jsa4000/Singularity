import numpy as np
import singularity as S
from singularity.components.layers import *
from singularity.components.optimizers import *
from singularity.components.regularizers import  *
from singularity.components.models import *

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

modeltype = 0

if (modeltype == 0):

    inputs = InputLayer((None,28*28))
    layer1 = DenseLayer(512, inputs = inputs,  activation = S.relu, init_weights = S.glorot_uniform)
    #layer1 = DenseLayer(512, input_shapes = (None,28*28),  activation = S.relu, init_weights = S.glorot_uniform)
    layer2 = DenseLayer(512, inputs = layer1,  activation = S.relu, init_weights = S.glorot_uniform)
    layer3 = DropoutLayer(0.2, inputs = layer2)
    outputs = DenseLayer(10, inputs = layer3,  activation = S.softmax, init_weights = S.glorot_uniform)
    model = GraphModel(inputs, outputs)
    #model = GraphModel(outputs = outputs)
else:
    model = OverlayModel()
    model.add(InputLayer((None,28*28)))
    model.add(DenseLayer(512, activation = S.relu, init_weights = S.glorot_uniform))
    #model.add(DenseLayer(512,input_shapes = (None,28*28), activation = S.relu, init_weights = S.glorot_uniform))
    model.add(DenseLayer(512, activation = S.relu, init_weights = S.glorot_uniform))
    model.add(DropoutLayer(0.2))
    model.add(DenseLayer(10, activation = S.softmax, init_weights = S.glorot_uniform))

rmsprop = RMSProp(learning_rate = 0.001)
model.build(loss=S.categorical_cross_entropy, optimizer = rmsprop, regularizers = [L2()])

model.train(X_train,Y_train, batch_size= batch_size, iterations = 1)
accuracy = model.accuracy(X_test,np.argmax(Y_test, axis=1))
print (accuracy)
 

