import numpy as np
from .exceptions import *

##############
# Base Class #
##############

class BaseClass (object):
    #constructor
    def __init__(self, **kwargs):
        self._attrs = {} 
        for key in kwargs:
            self._attrs[key] = kwargs[key]
            setattr(self, "_{}".format(key), self._attrs[key]) 
    ################################################
    @property
    def attrs(self):
        for key in self._attrs.keys():
            self._attrs[key] = getattr(self,"_{}".format(key))
        return self._attrs   
    ################################################
    #publicmethod
    def set_attr(self,key,item):
        self._attrs[key] = item
        setattr(self, "_{}".format(key), self._attrs[key]) 
    #publicmethod
    def del_attr(self,key):
        del self._attrs[key]
        delattr(self, "_{}".format(key)) 

#####################
# Commons Functions #
#####################

def get_padded_shape(x_shape,offset = 1):
    return tuple( item + (offset* 2) if (index + 2 >= len(x_shape)) else item for index,item in enumerate(x_shape))

def get_flattened_shape(x_shape, dimensions = 2):
    x_shape = np.asarray(x_shape)
    o_shape = []
    o_shape.extend(x_shape[range(dimensions - 1)])
    o_shape.extend([np.prod(x_shape[(dimensions - 1):])])
    return tuple(o_shape)

def broadcasting(x, shape):
    shapeT = tuple(dim for dim in shape if dim is not None)
    xT = x.transpose(shapeT)
    new_shape = tuple (-1 if axis is None else x.shape[axis] for index, axis in enumerate(shape))
    return xT.reshape(new_shape)

def is_collection(x):
    if (isinstance(x, (list, tuple, dict, np.ndarray))):
        return True
    else:
        return False

def is_none_or_empty(x):
    if (x is None):
        return True
    elif is_collection(x) and not len(x):
        return True
    else:
        return False

def is_primitive(data):
    primitives = (int, str, float, long, complex, bytearray, buffer,unicode, bool, type, np.ndarray)
    if (isinstance(data, primitives) and not is_none_or_empty(data)):
        return True
    else:
        return False

def is_complex(data):
    if (is_collection(data)):
        if (isinstance(data,dict)):
            data = data.values()
        for item in data:
            if (not is_primitive(item)):
                return True
        return False
    elif (is_primitive(data)):
        return False
    else:
        return True

def extract_args_kwargs(locals):
    params = {}
    parameters = locals.keys()
    parameters.remove("self")
    for key in parameters:
        if (key == "kwargs"):
            params.update(locals.get(key)) 
        else:
            params[key] = locals.get(key)
    return params

def cast(data, type):
    data.__class__ = type
    return data

########################
# Statistics Functions #
########################

def categorical(y, classes = None):
    if not classes:
        classes = np.max(y)+1
    Y = np.zeros((len(y), classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)