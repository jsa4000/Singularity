from collections import OrderedDict
import numpy as np
import six.moves.cPickle as pickle
import theano.tensor as T
from theano.tensor.signal import downsample 
from theano.tensor.nnet import conv

import theano

#https://github.com/genekogan/ofxTSNE

# The intention with this class is to provide a Module with some funcionality working from which in the future will be a Package

# 1. Core: 

# 1.1 Functions: non-linearities, loss-functions, operations, etc..



# 1.2. Ininitializators: Uniform, Binomial, Zeros, Ones, Random, Gaussian, etc.. (Unsupervised)


    

# 1.3. Variables, functions, loops, placeholders...

##############################
############ TEST ############
##############################

# This is a TEST to check if the placeholder and the variables works properly when defining the function ad the model

#framework = 1
#W = variable(init_variable((2,2),mode = 'O'))
#def model (x):
#    return sigmoid(dot(x,W))

#def test ():
  
#    x = placeholder((2,2))
#    y = model(x)

#    func = theano.function([x],y)
    
#    input = init_variable((2,2),mode = 'O')
#    output = func(input)

#    print(y)

#test()

##############################
    

# 2. Layers: Dense, InputLayers/OutputLayers, RNN, Embeding, Conv1D, Conv2D/Deconv, Pool/Unpool, Noise,


    
##############################
############ TEST ############
##############################

## This is a TEST to check if the placeholder and the variables works properly when defining the function ad the model

#framework = 0

#def test ():
  
#    #denslayer = Dense(8, activation = sigmoid)
#    #denslayer.set_params(Dense(3))

#    #inputs = init_variable((1,3),mode='O')
#    #outputs = denslayer.forward(inputs)

#    # Creates the networks without using the sequence
#    #inputlayer = InputLayer((None,3))
#    #hiddenlayer = Dense(4,input_layer = inputlayer)
#    #outputlayer = Dense(2,input_layer = hiddenlayer,activation = softmax)

#    #Create linear function to test the data
#    inputlayer = InputLayer((None,3))
#    hiddenlayer = Dense(4,input_layer = inputlayer,activation = None)
#    outputlayer = Dense(2,input_layer = hiddenlayer,activation = None)

#    #Init variables
#    inputs = ones((3,3))
    
#    # Since the input have been already set into the contructor there are no need to pass them again into this call
#    inputlayer.set_params(init_params = ones, init_bias = zeros)
#    hiddenlayer.set_params(init_params = ones, init_bias = zeros)
#    outputlayer.set_params(init_params = ones, init_bias = zeros)

#    # Forward propagation
#    outputs = inputlayer.forward(inputs)
#    outputs = hiddenlayer.forward(outputs)
#    outputs = outputlayer.forward(outputs)

#    print (outputs)

#test()

# This is a TEST to check if the placeholder and the variables works properly when defining the function ad the model

#framework = 1


#def model (x, layers):
#    y = x
#    for layer in layers:
#        y = layer.forward(y)
#    return y
    

#def test ():
  
#    #denslayer = Dense(8, activation = sigmoid)
#    #denslayer.set_params(Dense(3))

#    #inputs = init_variable((1,3),mode='O')
#    #outputs = denslayer.forward(inputs)

#    # Creates the networks without using the sequence
#    #inputlayer = InputLayer((None,3))
#    #hiddenlayer = Dense(4,input_layer = inputlayer)
#    #outputlayer = Dense(2,input_layer = hiddenlayer,activation = softmax)
#    inputlayer = InputLayer((None,3))
#    hiddenlayer = Dense(4,input_layer = inputlayer,activation = None)
#    outputlayer = Dense(2,input_layer = hiddenlayer,activation = None)

  
#    # Since the input have been already set into the contructor there are no need to pass them again into this call
#    inputlayer.set_params(init_params = ones, init_bias = zeros)
#    hiddenlayer.set_params(init_params = ones, init_bias = zeros)
#    outputlayer.set_params(init_params = ones, init_bias = zeros)

#    x = placeholder((None, 3)) # Inputs
#    y = placeholder((None,2)) # outputs
 
#    # Forward propagation - #1
#    #y = inputlayer.forward(x)
#    #y = hiddenlayer.forward(y)
#    #y = outputlayer.forward(y)
   
#    # Forward propagation - #2
#    # layers = [inputlayer,hiddenlayer, outputlayer]
#    #y = model(x, layers)

#    # Forward propagation - #3
#    y = outputlayer.forward(hiddenlayer.forward(inputlayer.forward(x)))

#    predict = theano.function ([x],y)

#      #Init variables
#    inputs = ones((3,3))

#    outputs = predict(inputs)
    

#    print (outputs)

#test()
##############################

##############################
############ TEST ############
##############################

## This is a TEST to check if the placeholder and the variables works properly when defining the function ad the model

#framework = 0

#def test ():
  
#    denslayer = Dense(8, activation = sigmoid)
#    denslayer.set_params( Dense(3))

#    inputs = init_variable((1,3),mode='O')
#    outputs = denslayer.forward(inputs)

#    print (outputs)

#test()

##############################




# 3. Regularization: Noise, Dropout, L1, L2..


    
##############################
############ TEST ############
##############################

## This is a TEST to check if the placeholder and the variables works properly when defining the function ad the model

#framework = 1

#def model (x, layers):
#    y = x
#    for layer in layers:
#        y = layer.forward(y)
#    return y

#def test ():
  
#    #denslayer = Dense(8, activation = sigmoid)
#    #denslayer.set_params(Dense(3))

#    #inputs = init_variable((1,3),mode='O')
#    #outputs = denslayer.forward(inputs)

#    # Creates the networks without using the sequence
#    inputlayer = InputLayer((None,3))
#    hiddenlayer = Dense(4,input_layer = inputlayer)
#    dropoutLayer = Dropout(0.9,input_layer = hiddenlayer)
#    outputlayer = Dense(2,input_layer = dropoutLayer,activation = softmax)

#    #Init variables
#    inputs = ones((3,3))
    
#    # Since the input have been already set into the contructor there are no need to pass them again into this call
#    inputlayer.set_params()
#    hiddenlayer.set_params(init_params = uniform, init_bias = zeros)
#    dropoutLayer.set_params()
#    outputlayer.set_params(init_params = uniform, init_bias = zeros)

#    # Forward propagation
#    outputs = inputlayer.forward(inputs)
#    outputs = hiddenlayer.forward(outputs)
#    outputs = dropoutLayer.forward(outputs)
#    outputs = outputlayer.forward(outputs)

#    #print (outputs)

#    x = placeholder((None, 3)) # Inputs
#    y = placeholder((None,2)) # outputs
   
#     #Forward propagation - #2
#    layers = [inputlayer,hiddenlayer, dropoutLayer, outputlayer] 
#    y = model(x, layers)

#    predict = theano.function ([x],y)

#    #Init variables
#    inputs = ones((3,3))

#    outputs = predict(inputs)
  
#    print (outputs)

#test()

##############################


# 4. Shape: Flatten, Matrix2D, pad, etc..

# 5. Models: Deep Neural Networks, Recurrent Neural Network, Generators, Generational Adversarial Network, Autoencoders, Bolztman Machines



        
##############################
############ TEST ############
##############################

## This is a TEST to check if the placeholder and the variables works properly when defining the function ad the model

#framework = 1

#def test ():
  
#    # Creates the mode

#    dnn = DeepNetwork()
        
#    inputlayer = InputLayer((None,3))
#    hiddenlayer = Dense(4,input_layer = inputlayer,activation = None)
#    outputlayer = Dense(2,input_layer = hiddenlayer,activation = None)

#    dnn.add(inputlayer)
#    dnn.add(hiddenlayer)
#    dnn.add(outputlayer)

#    dnn.build(loss=sum_squared_error, init_params = ones, init_bias = zeros)

#    #Init variables
#    inputs = ones((3,3))
    
#    pred_outputs = dnn.predict(inputs)
 
#    print (pred_outputs)

#    target_outputs = ones((3,2))
#    cost = dnn.cost(pred_outputs,target_outputs)

#    print (cost)

#test()

##############################

# 6. Optimizers: SGD, Back propagation, Momentum, Adam, etc..

"""
- Good resources about the implementation and thery for these optimizers.

    https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM
    http://sebastianruder.com/optimizing-gradient-descent/

    How optmizers works:

    Optimizer are the algorithms that are going to be used in the back-propagation during the training phase.
    Basically it changes the way that weights and other params (bias, states, etc..) are going to be updates in order to get sooner to the final solution.
        -> That means the solution where the cost is the lowest so we have get to a local minima solutios.
        -> In order to get to the global minima there are other algorithms like mini-batch training, regularizers, etc.. so it's easy to get to a final solution que converge.


    Stochastic Gradient Descent works

"""


##############################
############ TEST ############
##############################

#framework = 1

#def test ():
  
#    # Creates the mode

#    dnn = DeepNetwork()
        
#    inputlayer = InputLayer((None,3))
#    hiddenlayer = Dense(4,input_layer = inputlayer,activation = None)
#    outputlayer = Dense(2,input_layer = hiddenlayer,activation = None)

#    dnn.add(inputlayer)
#    dnn.add(hiddenlayer)
#    dnn.add(outputlayer)

#    sgd = SGD()
#    dnn.build(loss=sum_squared_error, optimizer = sgd, init_params = ones, init_bias = zeros)

#    #Init variables
#    inputs = ones((3,3))
    
#    pred_outputs = dnn.predict(inputs)
 
#    print (pred_outputs)

#    target_outputs = ones((3,2))
#    cost = dnn.cost(pred_outputs,target_outputs)

#    print (cost)

#test()


#framework = 1

#def test ():
  
#    # Creates the mode

#    dnn = DeepNetwork()
        
#    inputlayer = InputLayer((None,32*32))
#    hiddenlayer = Dense(32*32,input_layer = inputlayer,activation = tanh)
#    outputlayer = Dense(10,input_layer = hiddenlayer,activation = softmax)

#    dnn.add(inputlayer)
#    dnn.add(hiddenlayer)
#    dnn.add(outputlayer)

#    sgd = SGD()
#    dnn.build(loss=categorical_cross_entropy_cost, optimizer = sgd, init_params = uniform, init_bias = random)

#    import load

#    # load data
#    x_train, t_train, x_test, t_test= load.cifar10(dtype=theano.config.floatX)
#    labels_test= np.argmax(t_test, axis=1) # if the test x1 = 0 0 1 0, then return the argument with the maximun, in this case 2

#    dnn.train(x_train,t_train, batch_size= 25, iterations = 5)
#    accuracy = dnn.accuracy(x_test,labels_test)
#    print (accuracy)

#test()


#framework = 1

#def test ():
  
#    # Creates the mode

#    model = DeepNetwork()
        
#    #inputlayer = Dense(32*32, input_shape = (None,32*32),activation = sigmoid)
#    #outputlayer = Dense(10, activation = softmax)

#    #dnn.add(inputlayer)
#    #dnn.add(outputlayer)

#    #model.add(InputLayer((None,32*32)))
#    #model.add(Dense(512, activation = relu))
#    #model.add(Dropout(0.2))
#    #model.add(Dense(512, activation = relu))
#    #model.add(Dropout(0.2))
#    #model.add(Dense(10, activation = softmax))

#    model.add(InputLayer((None,32*32)))
#    model.add(Dense(512, activation = relu))
#    model.add(Dropout(0.2))
#    model.add(Dense(10, activation = softmax))

#    adam = Adam(learning_rate = 0.02)
#    adadelta = AdaDELTA()
#    adagrad = Adagrad(learning_rate = 0.02)
#    sgd = SGD(learning_rate = 0.02)
#    rmsprop = RMSProp(learning_rate = 0.01)
#    model.build(loss=categorical_cross_entropy_cost, optimizer = adam, init_params = glorot_uniform, init_bias = zeros)

#    import load

#    # load data
#    x_train, t_train, x_test, t_test= load.cifar10(dtype=theano.config.floatX)
#    labels_test= np.argmax(t_test, axis=1) # if the test x1 = 0 0 1 0, then return the argument with the maximun, in this case 2
   
#    model.train(x_train,t_train, batch_size= 300, iterations = 30)
#    accuracy = model.accuracy(x_test,labels_test)
#    print (accuracy)

#test()


#framework = 1

#def test ():
  
#    # Creates the mode

#    model = DeepNetwork()
        
#    #inputlayer = Dense(32*32, input_shape = (None,32*32),activation = sigmoid)
#    #outputlayer = Dense(10, activation = softmax)

#    #dnn.add(inputlayer)
#    #dnn.add(outputlayer)

#    model.add(InputLayer((None,28*28)))
#    model.add(Dense(512, activation = relu))
#    #model.add(Dropout(0.2))
#    #model.add(Dense(512, activation = relu))
#    #model.add(Dropout(0.2))
#    model.add(Dense(10, activation = softmax))

#    adam = Adam(learning_rate = 0.02)
#    adadelta = AdaDELTA()
#    adagrad = Adagrad(learning_rate = 0.002)
#    sgd = SGD(learning_rate = 0.002)
#    rmsprop = RMSProp(learning_rate = 0.001)
#    model.build(loss=categorical_cross_entropy_cost, optimizer = rmsprop, init_params = glorot_uniform, init_bias = zeros)

#    from keras.datasets import mnist
#    from keras.utils import np_utils

#    batch_size = 128
#    nb_epoch = 20

#    # the data, shuffled and split between train and test sets
#    (X_train, y_train), (X_test, y_test) = mnist.load_data()

#    X_train = X_train.reshape(60000, 784)
#    X_test = X_test.reshape(10000, 784)
#    X_train = X_train.astype('float32')
#    X_test = X_test.astype('float32')
#    X_train /= 255
#    X_test /= 255
#    print(X_train.shape[0], 'train samples')
#    print(X_test.shape[0], 'test samples')

#    # convert class vectors to binary class matrices
#    Y_train = np_utils.to_categorical(y_train, 10)
#    Y_test = np_utils.to_categorical(y_test, 10)

#    model.train(X_train,Y_train, batch_size= batch_size, iterations = 1)
#    accuracy = model.accuracy(X_test,np.argmax(Y_test, axis=1))
#    print (accuracy)
    
#    model.save("mnist.pkz")
    
#    # Loads te model 
#    modelLoaded = Model.load("mnist.pkz")
#    accuracy = modelLoaded.accuracy(X_test,np.argmax(Y_test, axis=1))
#    print (accuracy)


#test()


#framework = 1

#def test ():
 

#    from keras.datasets import mnist
#    from keras.utils import np_utils

#    # the data, shuffled and split between train and test sets
#    (X_train, y_train), (X_test, y_test) = mnist.load_data()

#    X_train = X_train.reshape(60000, 784)
#    X_test = X_test.reshape(10000, 784)
#    X_train = X_train.astype('float32')
#    X_test = X_test.astype('float32')
#    X_train /= 255
#    X_test /= 255

#    # convert class vectors to binary class matrices
#    Y_train = np_utils.to_categorical(y_train, 10)
#    Y_test = np_utils.to_categorical(y_test, 10)


#    # Loads te model 
#    modelLoaded = Model.load("mnist.pkz")
#    #accuracy = modelLoaded.accuracy(X_test,np.argmax(Y_test, axis=1))
#    #print (accuracy)

#    index = 555
#    test = X_test[index]
#    labelTest = Y_test[index]
#    pred = modelLoaded.predict(test[None,:])
#    print (pred)
    
#    import matplotlib.pyplot as plt
    
#    image =  np.asarray(test)
#    width = np.sqrt(np.sum(image.shape))
#    image = image.reshape(width,width)
#    plt.title('The result is: %s' %(pred[0]))
#    plt.axis('off')
#    plt.imshow(image, interpolation='nearest', cmap=plt.cm.binary)
#    plt.show()   


        

#test()


#framework = 1

#def test ():
  
#    # Creates the mode

#    model = DeepNetwork()
        
#    #inputlayer = Dense(32*32, input_shape = (None,32*32),activation = sigmoid)
#    #outputlayer = Dense(10, activation = softmax)

#    #dnn.add(inputlayer)
#    #dnn.add(outputlayer)

#    model.add(InputLayer((None,28*28)))
#    model.add(Dense(512, activation = relu))
#    #model.add(Dropout(0.2))
#    #model.add(Dense(512, activation = relu))
#    #model.add(Dropout(0.2))
#    model.add(Dense(10, activation = softmax))

#    adam = Adam(learning_rate = 0.02)
#    adadelta = AdaDELTA()
#    adagrad = Adagrad(learning_rate = 0.002)
#    sgd = SGD(learning_rate = 0.002)
#    rmsprop = RMSProp(learning_rate = 0.001)
#    #model.build(loss=categorical_cross_entropy_cost, optimizer = rmsprop, init_params = glorot_uniform, init_bias = zeros, regularizers = [L1(0.001),L2(0.0001)])
#    model.build(loss=categorical_cross_entropy_cost, optimizer = rmsprop, init_params = glorot_uniform, init_bias = zeros, regularizers = [L2()])

#    from keras.datasets import mnist
#    from keras.utils import np_utils

#    batch_size = 128
#    nb_epoch = 20

#    # the data, shuffled and split between train and test sets
#    (X_train, y_train), (X_test, y_test) = mnist.load_data()

#    X_train = X_train.reshape(60000, 784)
#    X_test = X_test.reshape(10000, 784)
#    X_train = X_train.astype('float32')
#    X_test = X_test.astype('float32')
#    X_train /= 255
#    X_test /= 255
#    print(X_train.shape[0], 'train samples')
#    print(X_test.shape[0], 'test samples')

#    # convert class vectors to binary class matrices
#    Y_train = np_utils.to_categorical(y_train, 10)
#    Y_test = np_utils.to_categorical(y_test, 10)

#    model.train(X_train,Y_train, batch_size= batch_size, iterations = 1)
#    accuracy = model.accuracy(X_test,np.argmax(Y_test, axis=1))
#    print (accuracy)
    
#    #model.save("mnist.pkz")
    
#    ## Loads te model 
#    #modelLoaded = Model.load("mnist.pkz")
#    #accuracy = modelLoaded.accuracy(X_test,np.argmax(Y_test, axis=1))
#    #print (accuracy)


#test()


framework = 1

def test ():
  
    # Creates the mode
    
    from keras.datasets import mnist
    from keras.utils import np_utils

    batch_size = 128
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
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype(E.floatX())
    X_test = X_test.astype(E.floatX())
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = DeepNetwork()

    model.add(InputLayer((None, 1, img_rows, img_cols)))
    model.add(Conv2D(nb_filters,(nb_conv, nb_conv), padding = 0, activation = relu))
    model.add(Conv2D(nb_filters,(nb_conv, nb_conv), padding = 0, max_pool_shape = (nb_pool, nb_pool), activation = relu))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation = relu))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation = softmax))
    
    adam = Adam(learning_rate = 0.02)
    adadelta = AdaDELTA()
    adagrad = Adagrad(learning_rate = 0.002)
    sgd = SGD(learning_rate = 0.002)
    rmsprop = RMSProp(learning_rate = 0.001)
    #model.build(loss=categorical_cross_entropy_cost, optimizer = rmsprop, init_params = glorot_uniform, init_bias = zeros, regularizers = [L1(0.001),L2(0.0001)])
    model.build(loss=categorical_cross_entropy_cost, optimizer = rmsprop, init_params = glorot_uniform, init_bias = zeros, regularizers = [L2()]) # L2 weight decay
       
    model.train(X_train,Y_train, batch_size= batch_size, iterations = 1)
    accuracy = model.accuracy(X_test,np.argmax(Y_test, axis=1))
    print (accuracy)
    
    #model.save("mnist.pkz")
    
    ## Loads te model 
    #modelLoaded = Model.load("mnist.pkz")
    #accuracy = modelLoaded.accuracy(X_test,np.argmax(Y_test, axis=1))
    #print (accuracy)


test()

# 7. Save Model and visualizing data