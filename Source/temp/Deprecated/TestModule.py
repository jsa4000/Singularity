import numpy as np
import singularity as S
from singularity.components import layers, optimizers, regularizers, models
from singularity.utils import datasets
from time import sleep
#import sys

#for i in range(21):
#    sys.stdout.write('\r')
#    # the exact output you're looking for:
#    sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
#    sys.stdout.flush()
#    sleep(0.25)
    
#def get_progress_bar(value, scale = 3):
#    return "[" + "#" * ((value / 10) * scale) + " " * ((scale * 10) - ((value / 10) * scale)) + "]"
  
#for item in range (100 + 1):
#    print "Percent complete", get_progress_bar(item), str(item) +"%","\r",
#    sleep(0.05)

#print "\nSigo con mas ejemplos "

#for item in range (1000):
#    print item," percent complete         \r",


#layer = layers.Dense(3)

#print 'Ejemplo de si funciona todo'
#print S.sqrt(2)

#x = S.random((3,4))
#print (x)
#y = S.argmax(x, axis = 1)
#print (y)

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

model = models.DeepNetwork()

model.add(layers.InputLayer((None, 1, img_rows, img_cols)))
model.add(layers.Convolution2D(nb_filters,(nb_conv, nb_conv), padding = 0, activation = S.relu))
model.add(layers.Convolution2D(nb_filters,(nb_conv, nb_conv), padding = 0,max_pool_shape = (nb_pool, nb_pool), activation = S.relu))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
#model.add(layers.Dense(128,activation = S.relu))
model.add(layers.Dense(144,activation = S.relu))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(nb_classes, activation = S.softmax))
    
adam = optimizers.Adam(learning_rate = 0.02)
adadelta = optimizers.AdaDELTA()
adagrad = optimizers.Adagrad(learning_rate = 0.002)
sgd = optimizers.SGD(learning_rate = 0.002)
rmsprop = optimizers.RMSProp(learning_rate = 0.001)
#model.build(loss=categorical_cross_entropy, optimizer = rmsprop, init_params = glorot_uniform, init_bias = zeros, regularizers = [L1(0.001),L2(0.0001)])
model.build(loss=S.categorical_cross_entropy, optimizer = sgd, init_params = S.glorot_uniform, init_bias = S.zeros, regularizers = [regularizers.L2()]) # L2 weight decay

#model.save("mnist_model.pkz",True)

## Loads te model 
#modelLoaded = models.Model.load("mnist_model.pkz.gz")
#accuracy = modelLoaded.accuracy(X_test,np.argmax(Y_test, axis=1))
#print (accuracy)

model.train(X_train,Y_train, batch_size= batch_size, iterations = 1)
accuracy = model.accuracy(X_test,np.argmax(Y_test, axis=1))
print (accuracy)
    
model.save("mnist_model.pkz")

Dense128_outputs = model.visualize_layer_prediction(S.broadcasting(X_test[0],(None,0,1,2)), 5 )
print (Dense128_outputs.shape)
print (Dense128_outputs)
    
import matplotlib.pyplot as plt
    
image =  Dense128_outputs[0]
width = np.sqrt(image.shape[0])
image = image.reshape(width,width)
#plt.title('The result is: %s' %(pred[0]))
plt.axis('off')
plt.imshow(image, interpolation='nearest', cmap=plt.cm.binary)
plt.show()   

  
    
## Loads te model 
#modelLoaded = Model.load("mnist.pkz")
#accuracy = modelLoaded.accuracy(X_test,np.argmax(Y_test, axis=1))
#print (accuracy)

