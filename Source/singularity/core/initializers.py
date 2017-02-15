from ..core import *

#########################
# Standard Initializers #
#########################

def ones (shape, dtype = floatX()):
    return np.ones(shape,dtype)    
  
def zeros (shape, dtype = floatX() ):
    return np.asarray(np.zeros(shape),dtype)
   
def random (shape, dtype = floatX()):
    return np.asarray(np.random.rand(*shape) * 0.1,dtype)

def uniform(shape, scale=0.05, dtype = floatX()):
    return np.asarray(np.random.uniform(low=-scale, high=scale, size=shape),dtype)

def normal(shape, scale=0.05, dtype = floatX()):
    return np.asarray(np.random.normal(loc=0.0, scale=scale, size=shape),dtype)

def binomial (x, p = 0.5,  dtype = floatX()):
    rp = 1 - p
    x *=  np.asarray(np.random.binomial(1, rp, np.prod(x.shape)).reshape(x.shape),dtype)
    return x / rp

########################
# Special Initializers #
########################

def lecun_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(3. / fan_in)
    return uniform(shape, scale)

def glorot_normal(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s)

def glorot_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s)

