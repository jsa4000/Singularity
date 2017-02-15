from collections import OrderedDict
import numpy as np
import six.moves.cPickle as pickle
import theano.tensor as T
from theano.tensor.signal import downsample 
from theano.tensor.nnet import conv

import theano

#https://github.com/genekogan/ofxTSNE

# The intention with this class is to provide a Module with some funcionality working from which in the future will be a Package

# First I need to define some basic funcionality for the Networks

# THe idea is to get a API that can run in different systems and backends

#Framework (0: Numpy, 1:Theano)
framework = 1

# 0. Exceptions and utils

class DeepPyException(Exception):
     def __init__(self, value):
         self.value = value
     def __str__(self):
         return repr(self.value)

def InitRandomize():
    if (framework == 0):
         return numpy.random.RandomState(1234)
    elif (framework == 1):
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        return RandomStreams(1234)
    else:
        raise DeepPyException("Framework doesn't found")

def InitFloatX():
    if (framework == 0):
         return np.float
    elif (framework == 1):
        return theano.config.floatX
    else:
        raise DeepPyException("Framework doesn't found")
    
floatX = theano.config.floatX
randomize = InitRandomize()

# 1. Core: 

# 1.1 Functions: non-linearities, loss-functions, operations, etc..

def softmax(x):
    """
    Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we 
    want to handle multiple classes. In logistic regression we assumed that the labels were binary: y(i)∈{0,1}. We used such 
    a classifier to distinguish between two kinds of hand-written digits. Softmax regression allows us to handle y(i)∈{1,…,K} 
    where K is the number of classes.
    """
    if (framework == 0):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)
    elif (framework == 1):
        return T.nnet.softmax(x)
    else:
        raise DeepPyException("Framework doesn't found")

def sigmoid(x):
    if (framework == 0):
        return 1.0 / (1.0 + np.exp(-x))
    elif (framework == 1):
        return T.nnet.sigmoid(x)
    else:
        raise DeepPyException("Framework doesn't found")

def tanh(x):
    if (framework == 0):
        return np.tanh(x)
    elif (framework == 1):
        return T.tanh(x)
    else:
        raise DeepPyException("Framework doesn't found")
    
def relu(x):
    """
        Function used due to the vanishing gradient problem in back-propagation.
    """
    if (framework == 0):
        y = x.copy()
        y[y < 0] = 0
        return y
    elif (framework == 1):
        return T.nnet.relu(x)
    else:
        raise DeepPyException("Framework doesn't found")  

def max(x):
    if (framework == 0):
       return np.max(x)
    elif (framework == 1):
        return T.max(x)
    else:
        raise DeepPyException("Framework doesn't found")  

def dot(x,y):
    if (framework == 0):
       return np.dot(x,y)
    elif (framework == 1):
        return T.dot(x,y)
    else:
        raise DeepPyException("Framework doesn't found")  

def sum(x):
    if (framework == 0):
       return np.sum(x)
    elif (framework == 1):
        return T.sum(x)
    else:
        raise DeepPyException("Framework doesn't found")  

def log(x):
    if (framework == 0):
       return np.log(x)
    elif (framework == 1):
        return T.log(x)
    else:
        raise DeepPyException("Framework doesn't found") 

def argmax(x, axis = 1):
    if (framework == 0):
       return np.argmax(x, axis)
    elif (framework == 1):
        return T.argmax(x,axis)
    else:
        raise DeepPyException("Framework doesn't found") 

def sqrt(x):
    if (framework == 0):
       return np.sqrt(x)
    elif (framework == 1):
        return T.sqrt(x)
    else:
        raise DeepPyException("Framework doesn't found") 

def power(x, p):
    if (framework == 0):
       return np.power(x,p)
    elif (framework == 1):
        return T.power(x,p)
    else:
        raise DeepPyException("Framework doesn't found") 

def get_padded_shape(x,offset = 1):
    return tuple( item + (offset* 2) if (index + 2 >= x.ndim) else item for index,item in enumerate(x.shape))

def padding (x, offset = 1):
    if (framework == 0):
        new_shape = tuple( item + (offset* 2) if (index + 2 >= x.ndim) else item for index,item in enumerate(x.shape))
        new_slice = tuple(slice(offset,-offset)  if (index + 2 >= len(new_shape)) else slice(None) for index,item in enumerate(new_shape) )
        zeromatrix = np.zeros(new_shape) 
        zeromatrix[new_slice] = x
        return zeromatrix
    elif (framework == 1):
        new_shape = tuple( item + (offset* 2) if (index + 2 >= x.ndim) else item for index,item in enumerate(x.shape) )
        new_slice = tuple(slice(offset,-offset)  if (index + 2 >= len(new_shape)) else slice(None) for index,item in enumerate(new_shape) )
        y = T.zeros(new_shape)
        return T.set_subtensor(y[new_slice], x)
    else:
        raise DeepPyException("Framework doesn't found") 

def get_flattened_shape(x_shape, dimensions = 2):
    x_shape = np.asarray(x_shape)
    o_shape = []
    o_shape.extend(x_shape[range(dimensions - 1)])
    o_shape.extend([np.prod(x_shape[(dimensions - 1):])])
    return tuple(o_shape)

def flatten(x, dimensions = 2):
    if (framework == 0):
        #Get the new shape once the layer has been flattened
        return x.reshape(tuple(get_flattened_shape(x.shape,dimensions)))
    elif (framework == 1):
        return x.flatten(dimensions)
    else:
        raise DeepPyException("Framework doesn't found") 

def dimshuffle(x, shape):
    """
        This function will allow the possibility to dimsuffle any matrix to the given shape (if it's possible)
        This function act like dimsuffle in Theano. Instead 'X' (theano) or -1 (numpy) a None value will be passed by value if new dimension.
        Parameters:
            x: matrix
                Matrix input that will be reshaped and transposed
            shape: tuple
                Shape for the desired new shape and transpose. None or np.newaxis will be used to add a new dimension to an axis.
    """
    if (framework == 0):
        #Get the new shape once the layer has been flattened
        shapeT = tuple(dim for dim in shape if dim is not None)
        xT = x.transpose(shapeT)
        new_shape = tuple (-1 if axis is None else x.shape[axis] for index, axis in enumerate(shape))
        return xT.reshape(new_shape)
    elif (framework == 1):
        new_shape = tuple ('x' if axis is None else axis for axis in shape)
        return x.dimshuffle(*new_shape)
    else:
        raise DeepPyException("Framework doesn't found") 

def conv2d(input, filter):
    if (framework == 0):
       raise DeepPyException("Not implemented") 
    elif (framework == 1):
        return conv.conv2d(input, filter)
    else:
        raise DeepPyException("Framework doesn't found") 

def max_pool_2d(input, maxpool_shape, ignore_border=True):
    if (framework == 0):
       raise DeepPyException("Not implemented") 
    elif (framework == 1):
        return downsample.max_pool_2d(input, maxpool_shape, ignore_border)
    else:
        raise DeepPyException("Framework doesn't found") 

########################################################
# In these cost functions the mean is already computed #
########################################################

def categorical_cross_entropy_cost (x, y, epsilon=1e-11 ):
    if (framework == 0):
       outputs = np.clip(x, epsilon, 1 - epsilon)
       return np.mean(-np.sum(y * np.log(outputs), axis=1))
    elif (framework == 1):
        return T.mean(T.nnet.categorical_crossentropy(x,y))
    else:
        raise DeepPyException("Framework doesn't found") 

def binary_cross_entropy_cost (x, y, epsilon=1e-11 ):
    if (framework == 0):
       outputs = np.clip(x, epsilon, 1 - epsilon)
       return np.mean(-np.sum(y * np.log(outputs) + (1 - y) * np.log(1 - outputs), axis=1))
    elif (framework == 1):
        return T.mean(T.nnet.binary_crossentropy(x,y))
    else:
        raise DeepPyException("Framework doesn't found") 

def sum_squared_error (x, y):
    if (framework == 0):
        return 0.5 * np.mean(np.sum(np.power(x - y,2), axis = 1 )) 
    elif (framework == 1):
        return  T.mean((x - y) ** 2)
    else:
        raise DeepPyException("Framework doesn't found") 

def negative_log_likelihood (x, y):
    if (framework == 0):
        return - np.mean(np.log(x)[np.asarray(range(y.shape[0]),y)])
    elif (framework == 1):
        return  -T.mean(T.log(x)[T.arange(y.shape[0]),y])
    else:
        raise DeepPyException("Framework doesn't found") 

# 1.2. Ininitializators: Uniform, Binomial, Zeros, Ones, Random, Gaussian, etc.. (Unsupervised)

def ones (shape, dtype = floatX):
    """ 
        Returns an array or Tensor with the shape defined.
        The function also take an object if already created previously        

        Parameters:
            shape: tuple
                Tuple with the shape of the new variable to create and initialize

            dtype: type as string
                if the input layer isn't specified then it will be inference from the outpu of the previous layer when the model will be build.
    """      
    return np.ones(shape,dtype)    
  
def zeros (shape, dtype = floatX ):
    """ 
        Returns an array or Tensor with the shape defined.
        The function also take an object if already created previously        

        Parameters:
            shape: tuple
                Tuple with the shape of the new variable to create and initialize

            dtype: type as string
                if the input layer isn't specified then it will be inference from the outpu of the previous layer when the model will be build.

    """
    return np.asarray(np.zeros(shape),dtype)
   
def random (shape, dtype = floatX):
    """ 
        Returns an array or Tensor with the shape defined.
        The function also take an object if already created previously        

        Parameters:
            shape: tuple
                Tuple with the shape of the new variable to create and initialize

            dtype: type as string
                if the input layer isn't specified then it will be inference from the outpu of the previous layer when the model will be build.
       
    """
    return np.asarray(np.random.rand(*shape) * 0.1,dtype)

def get_fans(shape, dim_ordering='th'):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if dim_ordering == 'th':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif dim_ordering == 'tf':
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def uniform(shape, scale=0.05):
    return np.random.uniform(low=-scale, high=scale, size=shape)


def normal(shape, scale=0.05):
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def lecun_uniform(shape, dim_ordering='th'):
    ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    '''
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    scale = np.sqrt(3. / fan_in)
    return uniform(shape, scale)


def glorot_normal(shape,  dim_ordering='th'):
    ''' Reference: Glorot & Bengio, AISTATS 2010
    '''
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s)

def glorot_uniform(shape, dim_ordering='th'):
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s)

def uniform (shape, dtype = floatX):
    """ 
        Returns an array or Tensor with the shape defined.
        The function also take an object if already created previously        

        Parameters:
            shape: tuple
                Tuple with the shape of the new variable to create and initialize

            dtype: type as string
                if the input layer isn't specified then it will be inference from the outpu of the previous layer when the model will be build.
     """
    return np.asarray(np.random.uniform(-np.sqrt(1./shape[0]), np.sqrt(1./shape[0]), shape),dtype)

def uniform (shape, dtype = floatX):
    """ 
        Returns an array or Tensor with the shape defined.
        The function also take an object if already created previously        

        Parameters:
            shape: tuple
                Tuple with the shape of the new variable to create and initialize

            dtype: type as string
                if the input layer isn't specified then it will be inference from the outpu of the previous layer when the model will be build.
     """
    return np.asarray(np.random.uniform(-np.sqrt(1./shape[0]), np.sqrt(1./shape[0]), shape),dtype)

def binomial (x, p = 0.5,  dtype = floatX):
    rp = 1 - p
    if (framework == 0):
        x *=  np.asarray(np.random.binomial(1, rp, np.prod(x.shape)).reshape(x.shape),dtype)
    elif (framework == 1):
        x *= randomize.binomial(x.shape, p=rp, dtype=dtype) # Generating a random mask to take random activations
    else:
        raise DeepPyException("Framework doesn't found")
    return x / rp
    

# 1.3. Variables, functions, loops, placeholders...

def gradient(cost, params):
    if (framework == 0):
       raise DeepPyException("Not implemented yet")
    elif (framework == 1):
        return theano.grad(cost, params)
    else:
        raise DeepPyException("Framework doesn't found")

def symbolicfunction(f):
    """
        This functiob will be used for functions that requies symbolic graphs to be compiled
    """
    def wrap(*args, **kw):
        if (framework == 0):
           #For numpy return only the function/s since there is non symbolic variables
           return f
        elif (framework == 1):
            return f(*args, **kw)
        else:
            raise DeepPyException("Framework doesn't found") 
    return wrap

def function(inputs, outputs, updates=None, givens=None, name=None):
    if (framework == 0):
       #For numpy return only the function/s since there is non symbolic variables
       return outputs
    elif (framework == 1):
        return theano.function(inputs,outputs, updates=updates, givens=givens, name=name)
    else:
        raise DeepPyException("Framework doesn't found") 

def placeholder (shape = (), dtype = floatX, ndim = None, name = None ):
    """
        Place holder to create tensor variables

        Parameters
        ----------
            shape: tuple
                Tuple with the shape of the current placeholder. Depending on the layer type the shape will be different. 
                    - () for scalars
                    - (10,) for vectors
                    - (1,3) for rows
                    - (12, 3) for Matrix
                    - (12, 3, 4 ,5) for Matrix4D
                    ...
            
            dtype: type as string
                if the input layer isn't specified then it will be inference from the outpu of the previous layer when the model will be build.
            
            ndim: integer
                Dimensions of the variable to be created
                With this attribute doesn't take into account all the possibles choices.
                    (See http://deeplearning.net/software/theano/library/tensor/basic.html)

            name: string
                Name for the current place holder
 
        Examples
        --------
            >> x = T.matrix()
            >> a = T.matrix()
            >> b = placeholder((3,3),dtype = 'float64')
            >> ia = T.ivector()
            >> ib = placeholder((1,3),dtype = 'int32')
            >> ic = placeholder((3,),dtype = 'int32')
            >> ic = placeholder(dtype = 'int32', ndim = 4)

            <TensorType(float64, matrix)>
            <TensorType(float64, matrix)>
            <TensorType(float64, matrix)>
            <TensorType(int32, vector)>
            <TensorType(int32, row)>
            <TensorType(int32, vector)>
            <TensorType(int32, 4D)>
    """        
    if (framework == 0):
        shape = np.asarray(shape)
        shape[shape == None] = 1
        return np.empty(tuple(shape),dtype)
    elif (framework == 1):
        if (shape is None and ndim is None):
            raise DeepPyException("Placeholder must have a dimension")
        elif shape is not None:
            shape = np.asarray(shape)
            shape[shape != 1] = 0
            shape = shape.astype(bool)
        else:
            shape = (False,)*ndim
        #Create the tensor type  
        """
            A _custom_shape variable could also be created dinamically to store the shape of the tensor
            t = T.TensorType(dtype=dtype, broadcastable=tuple(shape))(name)
            t._custom_shape = shape
        """
        return T.TensorType(dtype=dtype, broadcastable=tuple(shape))(name)
    else:
        raise DeepPyException("Framework doesn't found") 


def variable (object, dtype = floatX, broadcastable = None, name = None ):
    """ 
        Returns an array or Tensor with the type defined.
        If gpu is tru then the function will consider the best type in order to type the object for better GPU support depending on the device       
      
    """
    if (framework == 0):
       return np.asarray(object, dtype = dtype)
    elif (framework == 1):
        return theano.shared(np.asarray(object, dtype = dtype),broadcastable = broadcastable, name = name)
    else:
        raise DeepPyException("Framework doesn't found")

def constant (object, dtype = None, name = None):
    """ 
        Returns an array or Tensor with the type defined.
        If gpu is tru then the function will consider the best type in order to type the object for better GPU support depending on the device       
      
    """
    if (framework == 0):
       return object
    elif (framework == 1):
        return T.constant(1,dtype = dtype, name = name)
    else:
        raise DeepPyException("Framework doesn't found")

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

class Layer(object):
    def __init__(self, output_shape = None, input_shape = None, input_layer = None, activation = None, init_params = None, init_bias = None):
        """
        Base layer to build layers upon. This will be inherited for all the layers implemnented

        Parameters
        ----------
            output_shape: tuple
                Shape of the output that this layer will generate.  
            input_layer: a class "layer"
                For the first layer in the Network the input_layer or input_shape must be specified
                if the input layer isn't specified then it will be inferenced from the output of the previous layer when the model will be build.
            input_shape: tuple
                For the first layer in the Network the input_layer or input_shape must be specified (or InputLayer in sequence must be added)
                if the input layer isn't specified then it will be inferenced from the output of the previous layer when the model will be build.
            activation: function 
                This activation function will be call when generating the output of the layer with the shape specified. 
                If there is no function specified that means the function will be linear, so at then end there won't be any non-linearity function applied to the layer
            init_params,  init_bias: function
                Functions to define the initializations of the parameters and biases that will be created inside the layer
     
        Examples
        --------
      
        """        
        # Inititalize current Base class
        self._output_shape = output_shape
        self._input_layer = input_layer
        self._input_shape = input_shape
        self._activation = activation
        self._isInputLayer = False
        #Initialize initializers
        self._init_params = init_params
        self._init_bias = init_bias
        # Params of the layer
        self._params = {}
        self._trainableParams = []
        # Set initialized as false
        if (self._input_shape is not None):
            self._initialized = True
            self._isInputLayer = True
        else:
            self._initialized = False

    def set_params(self, input_layer = None, init_params = None, init_bias = None):
        # This method will initialize all the parameters needed for this layer (This will call by the model or sequence)
        if (input_layer is None and self._input_layer is None and self._input_shape is None ):
            raise DeepPyException("Input layer must be specified during the initialization of the Layer since it's the first one.") 
        elif (input_layer is not None):
            # Overrides the layer that was set at initialization 
            self._input_layer = input_layer
            #Get the previous shape using the current input layer
            self._input_shape = self._input_layer._output_shape
        elif (self._input_layer is not None):
                self._input_shape = self._input_layer._output_shape
        #Overrides initializers if configured  
        if (init_params is not None):
            self._init_params = init_params
        if (init_bias is not None):
            self._init_bias = init_bias
           
    def shape(self):
        if (self._output_shape is None):
            raise DeepPyException("Shape has not been defined for the current layer. Try to set the params using input_layer to conform.") 
        return self._output_shape

    def get_params(self, only_trainable = False):
        # return the params if initialized
        if (not self._initialized):
             raise DeepPyException("Layer must be initializted previously.")    
        if (only_trainable):
            return self._trainableParams
        else:    
            return self._params.values()
  
    def forward(self, inputs):
        # Check if current layer hsa been initialized
        if (not self._initialized):
            raise DeepPyException("Layer must be initializted previously.")
    
class InputLayer(Layer):
    def __init__(self, output_shape):
        """
        Input Layer for basic pourposes in Neural Networks
        
        Parameters
        ----------
            output_shape: tuple
                Shape of the input layer.  
     
        Examples
        --------
        
        """
        super(InputLayer,self).__init__(output_shape)
        # Initialize variables
        self._initialized = True
    
    def set_params(self, input_layer = None, init_params = None, init_bias = None):
        pass 
    def forward (self, inputs):
        return inputs
        
class Dense(Layer):
    def __init__(self, output_dim = None, output_shape = None, input_shape = None, input_layer = None, activation = tanh, init_params = random, init_bias = zeros):
        """
        Dense Layer for basic poposes in Neural Networks
        
        Parameters
        ----------
            output_tuple
                Shape of the output that this layer will generate.  

            See Layer parent of rthe resto of params
     
        Examples
        --------
            >> denslayer = Dense(8, activation = sigmoid)
            >> denslayer.set_params(Dense(3))

            # None means the input can be a list of values to train
            >> denslayer = Dense(8, input_shape = (None, 3), activation = sigmoid)
            >> denslayer.set_params()

            >> inputlayer = InputLayer((None,3))
            >> hiddenlayer = Dense(4,input_layer = inputlayer)
            >> outputlayer = Dense(2,input_layer = hiddenlayer,activation = softmax))

            >> inputs = init_variable((1,3),mode='O')
            >> outputs = denslayer.forward(inputs)

        """ 
        # Initialize variables
        if (output_dim is None and output_shape is None):
            raise DeepPyException("Outputs shape or dimension must be specified in Dense Layers.")   
        elif (output_shape is None):
            self._output_dim = output_dim
            self._output_shape = (None, self._output_dim)
        else:
            self._output_shape = output_shape

        # Call constructor
        super(Dense,self).__init__(self._output_shape, input_shape, input_layer, activation, init_params, init_bias )
       
    
    def set_params(self, input_layer = None, init_params = None, init_bias = None):
        super(Dense,self).set_params(input_layer,init_params, init_bias)
        # Create W parameters between this layer and the input layer -> (in, out)
        W = variable(self._init_params((self._input_shape[1],self._output_shape[1])))
        #Create bias paramters -> (out,) 
        b = variable(self._init_bias((self._output_shape[1],)))
        # Add to the params
        self._params = {'W': W, 'b':b}
        self._trainableParams.append(W)
        # Set initialized as true
        self._initialized = True
     
    def forward (self, inputs):
        # Check if the activation function was performed
        super(Dense,self).forward(inputs)
        if (self._isInputLayer):
            return inputs 
        # Perform de activation function 
        # For Dense layer activation (W * x + b)
        output = dot(inputs, self._params['W']) + self._params['b']
        if (self._activation is None):
            return output
        else:
            return self._activation (output)
    
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

class Padding(Layer):
    def __init__(self, padding = 1,input_shape = None, input_layer = None):
        """
        Input Layer for basic pourposes in Neural Networks
        
        Parameters
        ----------
            output_shape: tuple
                Shape of the input layer.  
     
        Examples
        --------
        
        """
        # Call constructor
        super(Flatten,self).__init__(input_shape = input_shape, input_layer = input_layer )
        # Initialize variables
        self._padding = padding
    
    def set_params(self, input_layer = None, init_params = None, init_bias = None):
        super(Flatten,self).set_params(input_layer)
        # There are no weights between this layer and the previous since        
         # This must set the output_shape the layer will have
        self._output_shape = get_padded_shape(self._input_shape, self._padding)
        # Set initialized as true
        self._initialized = True 
     
    def forward (self, inputs):
        # Return padding inputs
        # This will add an extra padding in the two last dimensions in inputs.
        return padding(inputs, self._padding)

class Flatten(Layer):
    def __init__(self, dimensions = 2, input_shape = None, input_layer = None):
        """
        Input Layer for basic pourposes in Neural Networks
        
        Parameters
        ----------
            output_shape: tuple
                Shape of the input layer.  
     
        Examples
        --------
        
        """
        # Call constructor
        super(Flatten,self).__init__(input_shape =  input_shape, input_layer = input_layer )
        # Initialize variables
        self._dimensions = dimensions
    
    def set_params(self, input_layer = None, init_params = None, init_bias = None):
        super(Flatten,self).set_params(input_layer)
        # There are no weights between this layer and the previous since        
        # This must set the output_shape the layer will have
        self._output_shape = get_flattened_shape(self._input_shape, self._dimensions)
        # Set initialized as true
        self._initialized = True 
    def forward (self, inputs):
        # Return flattened inputs
        return flatten(inputs, self._dimensions)

class Conv1D(Layer):
    pass

class Conv2D(Layer):
    def __init__(self, feature_maps = 4, filter_size = (3,3), max_pool_shape = None, padding = 0, 
                 input_shape = None, input_layer = None, activation = relu, init_params = random, init_bias = zeros):
        """
        Dense Layer for basic poposes in Neural Networks
        
        Parameters
        ----------
            output_tuple
                Shape of the output that this layer will generate.  

            See Layer parent of rthe resto of params
     
        Examples
        --------
    # dimshuffle: is like reshape in numpy. Where x means new col/row and 0 the current position of matrix[0,]
    # See more documentation about it
    
    # Creation of the Wigths and Biases for each layer

        w_c1 = init_weights((4, 1, 3, 3)) # 4 kernels with 3x3 size
        b_c1 = init_weights((4,))

        w_c2 = init_weights((8, 4, 3, 3))  # 8 kernels with 2x2 size, for each previous extracted feature
        b_c2 = init_weights((8,))

        w_h3 = init_weights((8 * 4 * 4, 100)) # Size of the final feauters extracted (using previous sizes). Size of the fully connected layer 
        b_h3 = init_weights((100,))

        w_o= init_weights((100, 10)) # Notmal weight of a NN, using the last unit's laters with the units for the next one. 
        b_o= init_weights((10,)) #The sum of the bias (1D) must coincide with the size of the colums (M) of the final matrix  (NxM)

    # First convolutional layer, between x and w_c1. A bias will be added to the result 
        c1 = T.maximum(0, conv2d(x, w_c1) + b_c1.dimshuffle('x', 0, 'x', 'x')) # RELU (CONV)
    # Max pool for the previous convolution layer with shape (3,3) -> DOWNSAMPLE
        p1 = max_pool_2d(c1, (3, 3))

    # Second convolutional layer, between last pool layer and w_c2. Also a bias will be added to the result 
        c2 = T.maximum(0, conv2d(p1, w_c2) + b_c2.dimshuffle('x', 0, 'x', 'x')) # RELU (CONV)
     # Max pool for the previous convolution layer with shape (2,2) -> Smaller that the first one -A
        p2 = max_pool_2d(c2, (2, 2))

    # Fully connected layer -> Also called Dense Layer
        p2_flat = p2.flatten(2) # flatter with two dimension (1000, 32*32) -> similar to the inputs when cifar was loaded

        p2_flat = dropout(p2_flat, p=p) # Add dropout

        h3 = T.maximum(0, T.dot(p2_flat, w_h3) + b_h3) # RELU (linear regression)

     # Last layer. A Softmax function will be computed using the previous output from the fully connected layer
        p_y_given_x= T.nnet.softmax(T.dot(h3, w_o) + b_o)

    # Finally return the value (0,1) because the sofmax function
         return p_y_given_x

        """ 
        # Initialize variables
        # In this part we need to check if all the parameters are valid

        # In the case of convolutional layers the output_shape will depend on the number of features, channels, filter size and padding, also the input shape will determine the output shape.
        # Also if the convolution layer has a max pooling layer, then this will also affect to the output shape.
        # In order to formalize the output shape (temporay), we will define a shape in 4D dimensions
        self._conv_shape = (feature_maps, None, filter_size[0], filter_size[1])
        self._max_pool_shape = max_pool_shape
        self._padding = padding

        # Would be interesing if a layer could be nested to another layer without the use of a model. For example max_pooling, zeropadding, etc.. 
                       
        # Call constructor
        super(Conv2D,self).__init__(input_shape = input_shape, input_layer = input_layer, activation = activation, 
                                    init_params = init_params, init_bias = init_bias )
       
    def set_params(self, input_layer = None, init_params = None, init_bias = None):
        super(Conv2D,self).set_params(input_layer,init_params, init_bias)
        #Get  the previous channels or feature maps for the previous layer and configure the filter for the convolution
        conv_shape = np.asarray(self._conv_shape)
        conv_shape[1] = np.asarray(self._input_shape)[1]
        self._conv_shape = tuple(conv_shape)
        # Create W parameters using the output size -> (feature_maps, channles, filter_size[0], filter_size[1])
        W = variable(self._init_params(self._conv_shape))
        #Create bias paramters -> (none, bias, none, none) 
        #b = variable(self._init_bias((None,self._conv_shape[0],None,None)))
        b = variable(self._init_bias((self._conv_shape[0],)))
        # Add to the params
        self._params = {'W': W, 'b':b}
        self._trainableParams.append(W)
        # Set the output shape for this layer depending on the configuration and the input_layer shape
        # Get the shape after the convolution. Only take one of the dimension for the filter size and the paddig applied
        reduction = self._padding - self._conv_shape[2] // 2 
        # This reduction will be applied for both dimensions width and height in the input shape
        self._output_shape = np.asarray(self._input_shape)
        # Instead channel we will have features maps from previous conv layer 
        self._output_shape[1] = self._conv_shape[0]
        # Apply the reduction
        self._output_shape[2:] = self._output_shape[2:] + (reduction * 2)
        # Finally if pool has been set then divide by the size of maxpool
        if (self._max_pool_shape is not None):
            self._output_shape[2:] = self._output_shape[2:] // self._max_pool_shape[0]
        #Set as tuble so it annot be modified
        self._output_shape = tuple(self._output_shape)
        # Set initialized as true
        self._initialized = True
     
    def forward (self, inputs):
        # Check if the activation function was performed
        super(Conv2D,self).forward(inputs)
        if (self._isInputLayer):
            return inputs 
        # Perform de activation function 
        # For Conv2d layer activation c1 = T.maximum(0, conv2d(x, w_c1) + b_c1.dimshuffle('x', 0, 'x', 'x')) # RELU (CONV)
        #output = conv2d(inputs, self._params['W']) + self._params['b'].dimshuffle('x', 0, 'x', 'x')
        output = conv2d(inputs, self._params['W']) + dimshuffle(self._params['b'],(None, 0, None, None))
        # Check if it's needed to apply an activation function
        if (self._activation is not None):
            output = self._activation (output)
        # Check if it's needed to do a max pool 2d
        if (self._max_pool_shape is not None):
            output = max_pool_2d(output, self._max_pool_shape)
        return output
  
# 3. Regularization: Noise, Dropout, L1, L2..

def L1(lambda_p = 0.001):
    def wrap(params):
        return (sum(list(abs(param).sum() for param in params))) * lambda_p
    return wrap

def L2(lambda_p = 0.0001):
    def wrap(params):
        return sum(list(power(param,2).sum() for param in params)) * lambda_p
    return wrap

class Dropout(Layer):
    def __init__(self, p = 0.5, input_shape = None, input_layer = None):
        """
        Input Layer for basic pourposes in Neural Networks
        
        Parameters
        ----------
            output_shape: tuple
                Shape of the input layer.  
     
        Examples
        --------
        
        """
        # Call constructor
        super(Dropout,self).__init__(self, input_shape, input_layer )
        # Initialize variables
        self._p = p
    
    def set_params(self, input_layer = None, init_params = None, init_bias = None):
        super(Dropout,self).set_params(input_layer)
        #Used the same output shape as the input layer
        self._output_shape = self._input_shape
        # Set initialized as true
        self._initialized = True 
     
    def forward (self, inputs):
        return binomial (inputs,self._p)
    
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

class Model (object):
    def __init__(self):
        """
        """
        # Layers that will compound the Network
        self._layers = []
        # Some variables where the principal component for the Training are stored.
        self._optimizer = None
        self._loss = None
        # Variable so get if the Model has been build
        self._compiled = False
        # Functions compiled
        self._predict = None
        self._train = None
        
    def count(self):
         return len(self._layers)
    def add(self, layer):
        if (isinstance(layer,Layer)):
            self._layers.append(layer)
        else:
            raise DeepPyException("Elements added must inherit from the base Layer clas.")  
    def insert(self, i, layer):
        if (isinstance(layer,Layer)):
            self._layers.insert(i,InputLayer)
        else:
            raise DeepPyException("Elements added must inherit from the base Layer class.")   
    def removeAt (self, i):
        if (len(self._layers) > i):
            layer = self._layers[i]
            self._layers.remove(layer)
        else:
            raise DeepPyException("Index higher than the elements in the model.")   
    def remove (self, layer):
        if (layer in self._layers):
            self._layers.remove(layer)
        else:
            raise DeepPyException("Layer not founded in the model.")   
    def extend(self, layers):
        self._layers.extend(layers)       
               
    def build(self, loss = None, optimizer = None, init_params = None, init_bias = None, regularizers = []):
        """
            When the Model has been set, this method must be called in order to build the model and set all the parameters, loss functions and optimizers.
           
            This is something very generic for a Model u Neoural Networks.

        """
        if (not len(self._layers)):
            raise DeepPyException("Model has no layers yet.") 

    def train (self):
        pass

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        return pickle.load(open(filename))
     
    def params (self, only_trainable = False):
        if (not len(self._layers)):
            raise DeepPyException("Model has no layers yet.") 
        params = []
        for layer in self._layers:
            params.extend(layer.get_params(only_trainable))    
        return params

    def accuracy (self, x_test, y_test):
        if (self._compiled):
            return self._train(inputs, outputs)
        else:
            raise DeepPyException("the model has not been compiled yet")   

    def predict (self, inputs):
        if (self._compiled):
            return self._predict(inputs)
        else:
            raise DeepPyException("the model has not been compiled yet")   

class DeepNetwork (Model):
    def __init__(self):
        super(DeepNetwork,self).__init__()
    def _forward_propagation (self, x):
        y = x
        for layer in self._layers:
            y = layer.forward(y)
        return y

    def _forward_propagation_no_dropout (self, x):
        y = x
        for layer in self._layers:
            if (not isinstance(layer,Dropout)):
                y = layer.forward(y)
        return y

    def _get_regularization (self, regularizers):
        return sum(list(regularizer(self.params(True)) for regularizer in regularizers ))
  
    def build(self, loss = categorical_cross_entropy_cost, optimizer = None, init_params = None, init_bias = None, regularizers = []):
        """
            This Method will build the DeepNetwork for a sequence of layers
        """
        #Call to the base layer
        super(DeepNetwork,self).build(loss, optimizer, init_params, init_bias)
        
        # First set and initialize all the params for the layers confifured (Skip the initialization of the input layer).
        for inputlayer, outputlayer in zip(self._layers,self._layers[1:]):
            outputlayer.set_params(input_layer = inputlayer,init_params = init_params, init_bias = init_bias)
        # Build the model using forward propagation
        #Create the input and output place holders depending on the size of each corresponding layer
        x = placeholder(self._layers[0].shape())
        t = placeholder(self._layers[-1].shape())
        # Generate the forward function usign all layers. y = self._forward_propagation(x) <- Not supported in numpy because symbolic variables are evaluated after the compilation
        p_y_given_x = symbolicfunction(self._forward_propagation)(x)
        p_y_given_x_ndo = symbolicfunction(self._forward_propagation_no_dropout)(x)
        y = argmax(p_y_given_x_ndo, axis=1)
        cost = symbolicfunction(loss)(p_y_given_x,t) + self._get_regularization(regularizers)
        # Compile the function to predict the result 
        self._predict = function([x],y)
        # Get the optimizations with the updates
        updates = optimizer.optimize(cost = cost, params = self.params())
        #Compile the function ot predict the cost
        self._train = function([x,t],cost,updates = updates)
        # Finally set compiled true
        self._compiled = True
        #print "Prediction"
        #theano.printing.pprint(pred)
        #print "Cost"
        #theano.printing.pprint(cost)
      
    def train (self,x_train, t_train, batch_size= 50, iterations = 3):
        for i in range(iterations):
            print "iteration %d" % (i+ 1)
            for start in range(0, len(x_train), batch_size):
                x_batch= x_train[start:start+ batch_size]
                t_batch= t_train[start:start+ batch_size]
                cost = self._train(x_batch, t_batch)
                print "cost: %.5f" % cost  

    def accuracy (self,x_test, y_test ):
        predictions_test= self._predict(x_test)
        accuracy = np.mean(predictions_test== y_test)
        print "accuracy: %.5f" % accuracy
        return  accuracy

        
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


class Optimizer (object):
    def __init__(self):
       pass
    def optimize(self, cost, params ):
        pass

class SGD (Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super(SGD,self).__init__()
        self._learning_rate = learning_rate
        self._momentum = momentum

    def optimize(self, cost, params ):
        grads = gradient(cost, params)
        updates = []
        for p, g in zip(params, grads):
            mparam_i = variable(zeros(p.get_value().shape))
            v = self._momentum * mparam_i - self._learning_rate * g
            updates.append((mparam_i, v))
            updates.append((p, p + v))
        return updates

class RMSProp (Optimizer):
    """
        http://sebastianruder.com/optimizing-gradient-descent/
        http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(self, learning_rate=0.01, rho=0.9, epsilon=1e-6):
        super(RMSProp,self).__init__()
        self.learning_rate = learning_rate
        self._rho = rho
        self._epsilon = epsilon

    def optimize(self, cost, params ):
        grads = gradient(cost, params)
        updates = OrderedDict()
         # Using theano constant to prevent upcasting of float32
        one = constant(1)

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            accu = variable(zeros(value.shape), dtype=value.dtype,
                                 broadcastable=param.broadcastable)
            accu_new = self._rho * accu + (one - self._rho) * grad ** 2
            updates[accu] = accu_new
            updates[param] = param - (self.learning_rate * grad /
                                      sqrt(accu_new + self._epsilon))

        return updates

class Adagrad (Optimizer):
    """
       http://sebastianruder.com/optimizing-gradient-descent/ 
       https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-6):
        super(Adagrad,self).__init__()
        self.learning_rate = learning_rate
        self._epsilon = epsilon

    def optimize(self, cost, params ):
        grads = gradient(cost, params)
        updates = OrderedDict()
         # Using theano constant to prevent upcasting of float32
        one = constant(1)

        for param, grad in zip(params, grads):
            value = param.get_value(borrow=True)
            accu = variable(zeros(value.shape), dtype=value.dtype,
                            broadcastable=param.broadcastable)
            accu_new = accu + grad ** 2
            updates[accu] = accu_new
            updates[param] = param - ((self.learning_rate * grad) /
                                      (sqrt(accu_new) + self._epsilon))
        return updates

class AdaDELTA (Optimizer):
    """
       http://sebastianruder.com/optimizing-gradient-descent/
       https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM
    """
    def __init__(self, rho=0.9, epsilon=1e-6):
        super(AdaDELTA,self).__init__()
        self._rho = rho
        self._epsilon = epsilon

    def optimize(self, cost, params ):
        grads = gradient(cost, params)
        updates = OrderedDict()
         # Using theano constant to prevent upcasting of float32
        one = constant(1)

        for param, grad in zip(params, grads):
            value = param.get_value(borrow = True)
            # Create shared variables updates with previous states
            accu = variable(zeros(value.shape), dtype=value.dtype,
                            broadcastable=param.broadcastable)
            delta_accu = variable(zeros(value.shape), dtype=value.dtype,
                            broadcastable=param.broadcastable)
            #Compute the uptades based on AdaDELTA 
            accu_new = self._rho * accu + (one - self._rho) * (grad ** 2 )
            update = - sqrt (delta_accu + self._epsilon) * grad / sqrt (accu_new + self._epsilon)
            delta_accu_new = self._rho * delta_accu + (one - self._rho) * (update ** 2)
                   
            # Store the updates
            updates[accu] = accu_new
            updates[delta_accu] = delta_accu_new
            updates[param] = param + update
        return updates

class Adam (Optimizer):
    """
        http://sebastianruder.com/optimizing-gradient-descent/
        https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM
    """
    def __init__(self, learning_rate=0.002, beta1=0.9,beta2=0.999, epsilon=1e-8):
        super(Adam,self).__init__()
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def optimize(self, cost, params ):
        grads = gradient(cost, params)
        updates = OrderedDict()
        
        # Using theano constant to prevent upcasting of float32
        one = constant(1)
        t_prev = variable(0.)

        t = t_prev + 1
        a_t = self._learning_rate * sqrt(one - self._beta2**t)/(one - self._beta1**t)

        for param, grad in zip(params, grads):
            value = param.get_value(borrow = True)
            # Create shared variables updates with previous states
            accu = variable(zeros(value.shape), dtype=value.dtype,
                            broadcastable=param.broadcastable)
            moment = variable(zeros(value.shape), dtype=value.dtype,
                            broadcastable=param.broadcastable)

            #Compute the uptades based on AdaDELTA 
            moment_new = self._beta1 * moment + (one - self._beta1) * grad 
            accu_new = self._beta2 * accu + (one - self._beta2) * (grad ** 2 )
          
            update = a_t * moment_new / (sqrt(accu_new) + self._epsilon)
                             
            # Store the updates
            updates[moment] = moment_new
            updates[accu] = accu_new
            updates[param] = param - update
            
        updates[t_prev] = t
        return updates

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
    X_train = X_train.astype(floatX)
    X_test = X_test.astype(floatX)
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