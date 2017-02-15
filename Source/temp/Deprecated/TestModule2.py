import numpy as np
import singularity as S
from singularity.components import layers, optimizers, regularizers, models
from singularity.utils import datasets
from time import sleep
import matplotlib.pyplot as plt

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


############
# COMMMENTED
############

model = models.DeepNetwork()

model.add(layers.Input((None,28*28)))
#model.add(layersDense(512, activation = relu))
model.add(layers.Dense(484, activation = S.relu))
model.add(layers.Dropout(0.2))
#model.add(Dense(512, activation = relu))
#model.add(Dropout(0.2))
model.add(layers.Dense(10, activation = S.softmax))

adam = optimizers.Adam(learning_rate = 0.02)
adadelta = optimizers.AdaDELTA()
adagrad = optimizers.Adagrad(learning_rate = 0.002)
sgd = optimizers.SGD(learning_rate = 0.002)
rmsprop = optimizers.RMSProp(learning_rate = 0.001)
#model.build(loss=categorical_cross_entropy, optimizer = rmsprop, init_params = glorot_uniform, init_bias = zeros, regularizers = [L1(0.001),L2(0.0001)])
model.build(loss=S.categorical_cross_entropy, optimizer = rmsprop, init_params = S.glorot_uniform, init_bias = S.zeros, regularizers = [regularizers.L2()])

model.train(X_train,Y_train, batch_size= batch_size, iterations = 1)
accuracy = model.accuracy(X_test,np.argmax(Y_test, axis=1))
print (accuracy)
    
model.save("mnist_log_model.pkz")
    
# Loads te model 
model = models.Model.load("mnist_log_model.pkz")
accuracy = model.accuracy(X_test,np.argmax(Y_test, axis=1))
print (accuracy)
    
param = model.get_layer_params(1)[0]
print (param.get_value(borrow=True).shape)

value = param.get_value(borrow=True)

image = value.reshape(value.shape[0],value.shape[1])
#plt.title('The result is: %s' %(pred[0]))
plt.axis('off')
plt.imshow(image, interpolation='nearest', cmap=plt.cm.binary)
plt.show()   

############
# COMMMENTED
############

# Loads te model 
model = models.Model.load("mnist_log_model.pkz")
    
param = model.get_layer_params(1)[0]
print (param.get_value(borrow=True).shape)

value = param.get_value(borrow=True)
image = value[:,5]
print (image.shape[0])
width = np.sqrt(image.shape[0])
image = image.reshape(width,width)

#plt.title('The result is: %s' %(pred[0]))
plt.axis('off')
plt.imshow(image, interpolation='nearest', cmap=plt.cm.binary)
plt.show()   


print ("done")


#Dense128_outputs = model.visualize_layer_prediction(S.broadcasting(X_test[0],(None,0)), 1 )
#print (Dense128_outputs.shape)
    
#import matplotlib.pyplot as plt
    
#image =  Dense128_outputs[0]
#width = np.sqrt(image.shape[0])
#image = image.reshape(width,width)
##plt.title('The result is: %s' %(pred[0]))
#plt.axis('off')
#plt.imshow(image, interpolation='nearest', cmap=plt.cm.binary)
#plt.show()   

  
    
## Loads te model 
#modelLoaded = Model.load("mnist.pkz")
#accuracy = modelLoaded.accuracy(X_test,np.argmax(Y_test, axis=1))
#print (accuracy)

