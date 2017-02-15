from .commons import *
from .exceptions import *
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

##############################
# Initialze global variables #
##############################

def randomize(seed = 1234):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    return RandomStreams(seed)
  
def floatX():
    return theano.config.floatX

#########################
# Activations Functions #
#########################

def softmax(x):
    return T.nnet.softmax(x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def tanh(x):
    return T.tanh(x)
   
def relu(x):
    return T.nnet.relu(x)

#######################
# Statistic Functions #
#######################

def max(x):
    return T.max(x)

def exp(x):
    return T.exp(x)

def dot(x,y):
    return T.dot(x,y)
 
def sum(x):
    return T.sum(x)

def log(x):
     return T.log(x)

def argmax(x, axis = 1):
    return T.argmax(x,axis)

def mean(x, axis = 1):
    return T.mean(x, axis)

def sqrt(x):
    return T.sqrt(x)

def power(x, p):
    return T.power(x,p)

def square(x):
    return T.power(x,2)

##############################
# Special Matrices Functions #
##############################

def padding (x, offset = 1):
    new_shape = tuple( item + (offset* 2) if (index + 2 >= x.ndim) else item for index,item in enumerate(x.shape) )
    new_slice = tuple(slice(offset,-offset)  if (index + 2 >= len(new_shape)) else slice(None) for index,item in enumerate(new_shape) )
    y = T.zeros(new_shape)
    return T.set_subtensor(y[new_slice], x)

def flatten(x, dimensions = 2):
    return x.flatten(dimensions)

def dimshuffle(x, shape):
    new_shape = tuple ('x' if axis is None else axis for axis in shape)
    return x.dimshuffle(*new_shape)

def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        field_size = np.prod(shape[2:])
        fan_in = shape[1] * field_size
        fan_out = shape[0] * field_size
    else:
        # no specific assumptions
        fan_in = T.sqrt(T.prod(shape))
        fan_out = T.sqrt(T.prod(shape))
    return fan_in, fan_out

def convolution2d(x, filter):
    return conv.conv2d(x, filter)

def max_pool_2d(x, pool_shape, ignore_border=True):
    return pool.pool_2d(x, pool_shape, ignore_border)
   
#####################
# Random Generators #
#####################
  
def random_binomial (x, p = 0.5,  dtype = floatX()):
    rp = 1 - p
    x *= randomize().binomial(x.shape, p=rp, dtype=dtype) # Generating a random mask to take random activations
    return x / rp

def random_normal(shape, mean=0.0, std=1.0, dtype=floatX()):
    return randomize().normal(size=shape, avg=mean, std=std, dtype=dtype)

##################
# Loss Functions #
##################

def categorical_cross_entropy(x, y, epsilon=1e-11 ):
    return T.mean(T.nnet.categorical_crossentropy(x,y))
  
def binary_cross_entropy (x, y, epsilon=1e-11 ):
    return T.mean(T.nnet.binary_crossentropy(x,y))
   
def sum_squared_error (x, y):
    return  T.mean((x - y) ** 2)
 
def negative_log_likelihood (x, y):
    return  -T.mean(T.log(x)[T.arange(y.shape[0]),y])

######################################
# Symbolic Definitions and Functions #
######################################

def gradient(cost, params):
    return theano.grad(cost, params)
 
def symbolicfunction(f):
    def wrap(*args, **kw):
        return f(*args, **kw)
    return wrap

def function(inputs, outputs, updates=None, givens=None, name=None):
    return theano.function(inputs,outputs, updates=updates, givens=givens, name=name)
   
def placeholder (shape = (), dtype = floatX(), ndim = None, name = None ):
    if (shape is None and ndim is None):
        raise singularity_exception("Placeholder must have a dimension")
    elif shape is not None:
        shape = np.asarray(shape)
        shape[shape != 1] = 0
        shape = shape.astype(bool)
    else:
        shape = (False,)*ndim
    return T.TensorType(dtype=dtype, broadcastable=tuple(shape))(name)

def variable (x, dtype = floatX(), broadcastable = None, name = None ):
    return theano.shared(np.asarray(x, dtype = dtype),broadcastable = broadcastable, name = name)

def constant (x, dtype = None, name = None):
    return T.constant(x, dtype = dtype, name = name)

def get_value (x):
    return x.get_value()
  
def is_variable(data):
    if (isinstance(data, theano.compile.sharedvalue.SharedVariable)):
        return True
    return False   