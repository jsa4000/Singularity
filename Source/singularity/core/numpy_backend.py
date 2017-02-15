from .commons import *
from .exceptions import *

##############################
# Initialze global variables #
##############################

def randomize(seed = 1234):
    return np.random.RandomState(seed)
  
def floatX():
    return np.float
 
#########################
# Activations Functions #
#########################

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)
  
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
  
def tanh(x):
    return np.tanh(x)
    
def relu(x):
    y = x.copy()
    y[y < 0] = 0
    return y
 
#######################
# Statistic Functions #
#######################

def max(x):
    return np.max(x)

def exp(x):
    return np.exp(x)

def dot(x,y):
    return np.dot(x,y)
 
def sum(x):
    return np.sum(x)

def log(x):
    return np.log(x)

def argmax(x, axis = 1):
    return np.argmax(x, axis)

def mean(x, axis = 1):
    return np.mean(x, axis)

def sqrt(x):
    return np.sqrt(x)

def power(x, p):
    return np.power(x,p)

def square(x):
    return power(x,2)
  
##############################
# Special Matrices Functions #
##############################

def padding (x, offset = 1):
    new_shape = tuple( item + (offset* 2) if (index + 2 >= x.ndim) else item for index,item in enumerate(x.shape))
    new_slice = tuple(slice(offset,-offset)  if (index + 2 >= len(new_shape)) else slice(None) for index,item in enumerate(new_shape) )
    zeromatrix = np.zeros(new_shape) 
    zeromatrix[new_slice] = x
    return zeromatrix

def flatten(x, dimensions = 2):
    return x.reshape(tuple(get_flattened_shape(x.shape,dimensions)))
 
def dimshuffle(x, shape):
    shapeT = tuple(dim for dim in shape if dim is not None)
    xT = x.transpose(shapeT)
    new_shape = tuple (-1 if axis is None else x.shape[axis] for index, axis in enumerate(shape))
    return xT.reshape(new_shape)
  
def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        field_size = np.prod(shape[2:])
        fan_in = shape[1] * field_size
        fan_out = shape[0] * field_size
    else:
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def convolution2d(x, filter):
    raise exceptions.not_implemented_exception
   
def max_pool_2d(x, pool_shape, ignore_border=True):
    raise exceptions.not_implemented_exception

#####################
# Random Generators #
#####################
   
def random_binomial (x, p = 0.5, dtype = floatX()):
    rp = 1 - p
    x *=  np.asarray(np.random.binomial(1, rp, np.prod(x.shape)).reshape(x.shape),dtype)
    return x / rp

def random_normal(shape, mean=0.0, std=1.0, dtype=floatX()):
    return np.asarray(np.random.normal(loc=mean, scale=std, size=shape),dtype)

##################
# Loss Functions #
##################

def categorical_cross_entropy (x, y, epsilon=1e-11 ):
    outputs = np.clip(x, epsilon, 1 - epsilon)
    return np.mean(-np.sum(y * np.log(outputs), axis=1))
   
def binary_cross_entropy (x, y, epsilon=1e-11 ):
    outputs = np.clip(x, epsilon, 1 - epsilon)
    return np.mean(-np.sum(y * np.log(outputs) + (1 - y) * np.log(1 - outputs), axis=1))
   
def sum_squared_error (x, y):
    return 0.5 * np.mean(np.sum(np.power(x - y,2), axis = 1 )) 
  
def negative_log_likelihood (x, y):
    return - np.mean(np.log(x)[np.asarray(range(y.shape[0]),y)])

######################################
# Symbolic Definitions and Functions #
######################################

def gradient(cost, params):
    raise exceptions.not_implemented_exception
   
def symbolicfunction(f):
    def wrap(*args, **kw):
        return f
    return wrap

def function(inputs, outputs, updates=None, givens=None, name=None):
    return outputs
   
def placeholder (shape = (), dtype = floatX(), ndim = None, name = None ):
    shape = np.asarray(shape)
    shape[shape == None] = 1
    return np.empty(tuple(shape),dtype)

def variable (x, dtype = floatX(), broadcastable = None, name = None ):
     return np.asarray(x, dtype = dtype)
   
def constant (x, dtype = None, name = None):
    return x
    
def get_value (x):
    return x

def is_variable(data):
    if (isinstance(data, np.ndarray)):
        return True
    return False   