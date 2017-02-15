from .commons import *
from .exceptions import *

##############################
# Initialze global variables #
##############################

def randomize(seed = 1234):
    raise exceptions.not_implemented_exception
  
def floatX():
    raise exceptions.not_implemented_exception

#########################
# Activations Functions #
#########################

def softmax(x):
    raise exceptions.not_implemented_exception

def sigmoid(x):
   raise exceptions.not_implemented_exception

def tanh(x):
    raise exceptions.not_implemented_exception
   
def relu(x):
    return T.nnet.relu(x)

#######################
# Statistic Functions #
#######################

def max(x):
    raise exceptions.not_implemented_exception

def exp(x):
    raise exceptions.not_implemented_exception

def dot(x,y):
    raise exceptions.not_implemented_exception
 
def sum(x):
    raise exceptions.not_implemented_exception

def log(x):
     raise exceptions.not_implemented_exception

def argmax(x, axis = 1):
    raise exceptions.not_implemented_exception

def mean(x, axis = 1):
    raise exceptions.not_implemented_exception

def sqrt(x):
    raise exceptions.not_implemented_exception

def power(x, p):
    raise exceptions.not_implemented_exception

def square(x):
    raise exceptions.not_implemented_exception

##############################
# Special Matrices Functions #
##############################

def padding (x, offset = 1):
    raise exceptions.not_implemented_exception

def flatten(x, dimensions = 2):
    raise exceptions.not_implemented_exception

def dimshuffle(x, shape):
    raise exceptions.not_implemented_exception

def get_fans(shape):
    raise exceptions.not_implemented_exception 

def convolution2d(x, filter):
    raise exceptions.not_implemented_exception

def max_pool_2d(x, pool_shape, ignore_border=True):
    raise exceptions.not_implemented_exception
   
#####################
# Random Generators #
#####################
  
def random_binomial (x, p = 0.5,  dtype = floatX()):
    raise exceptions.not_implemented_exception

def random_normal(shape, mean=0.0, std=1.0, dtype=floatX()):
    raise exceptions.not_implemented_exception

##################
# Loss Functions #
##################

def categorical_cross_entropy (x, y, epsilon=1e-11 ):
    raise exceptions.not_implemented_exception
  
def binary_cross_entropy (x, y, epsilon=1e-11 ):
    raise exceptions.not_implemented_exception
   
def sum_squared_error (x, y):
    raise exceptions.not_implemented_exception
 
def negative_log_likelihood (x, y):
    raise exceptions.not_implemented_exception

######################################
# Symbolic Definitions and Functions #
######################################

def gradient(cost, params):
    raise exceptions.not_implemented_exception
 
def symbolicfunction(f):
    raise exceptions.not_implemented_exception

def function(inputs, outputs, updates=None, givens=None, name=None):
    raise exceptions.not_implemented_exception
   
def placeholder (shape = (), dtype = floatX(), ndim = None, name = None ):
    raise exceptions.not_implemented_exception

def variable (x, dtype = floatX(), broadcastable = None, name = None ):
    raise exceptions.not_implemented_exception  

def constant (x, dtype = None, name = None):
    raise exceptions.not_implemented_exception

def get_value (x):
    raise exceptions.not_implemented_exception

def get_variable_type ():
    raise exceptions.not_implemented_exception

def is_variable(data):
    raise exceptions.not_implemented_exception