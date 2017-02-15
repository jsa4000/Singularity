import numpy as np
import singularity as S
import inspect
from singularity.components.layers import *
from singularity.components.optimizers import *
from singularity.components.regularizers import  *
from singularity.components.models import *
from singularity.utils import datasets
from singularity.utils import hdf5

class AnotherClass (object):
    def __init__(self, name):
        self._name = name

# The main pourpose with this base class is to generalize the  process of manipulation of all main attributes and parameters

#########
# MyClass #
######### 

class MyClass (S.BaseClass):
    def __init__(self, inputs = None, outputs  = None, loss = None, optimizer = None, regularizers = [], name = None, **kwargs):
        ## First at all get locals from the current function
     
        params = S.extract_args_kwargs(locals())
        if ("inputs" in params):
            params["inputs"] = self._get_nodes(params["inputs"]) 

        super(MyClass, self).__init__(**params)

    def _get_nodes(self, nodes):
        if (nodes == None):
            result = []
        elif (isinstance(nodes,tuple)):
            result = []
            result.append(nodes)
        else:
            result = nodes
        return result 

    def somthing_to_do(self):
        return 4

class MyClass2 (MyClass):
    def __init__(self, age = None, node = None, **kwargs):
        params = S.extract_args_kwargs(locals())
        super(MyClass2, self).__init__(**params)

        

test = MyClass(name = "Javier", inputs = [(None,23),(None,65)], outputs = (5, 10))
print test._name
print test.attrs["name"]
#Can I change the paramters?
test._name = "Pedro"
print "Now I'm", test._name
print "Now I'm", test.attrs["name"]

test2 = MyClass2(age = 32, name = "Rodrigo", inputs = [(23,23), (None, 65)], outputs = (None, 10), node=test )
print test2._name
# Attributes are declared statics?
test2._name = "Marta"
print "Now I'm", test2._name
print "But I'm still calling me", test._name

name = "This is a string"
i = 100
v = variable([0,1,2,3,4,5])
d = {"name": "Javier", "lastname":"Santos Andr?s", "age":33}
t = (4,None, 5)

def sum (x,y):
    return x + y

func = sum

n = np.asarray(range(5), dtype = theano.config.floatX)
print (type(n).__name__)

p = np.asarray(range(3*4*2), dtype = 'int32').reshape(3,4,2)

node1 = {"name": "node1", "val":v, "param": [variable(n),variable(p)]}
node2 = {"name": "node2", "val":t, "param": {"W": variable(p), "b":variable(n)}}
nodes = {"node1":node1, "node2":node2}

m = {"name": "model", "nodes":nodes, "test":test, "another":AnotherClass("pedro"), "myfunc":func}


##Save the model into a hdf5 file format
#hdf5.save("test.hdf", m, "root")

##Load the data settings
#data = hdf5.load("test.hdf")
#print data["myfunc"](5,6)

#Save the model into a hdf5 file format
hdf5.save("test.hdf", test2, "root")

# Load the data settings
data = hdf5.load("test.hdf")

# Follwoing sentence is to cast an object into a derived one.
# From Base class to MyClass
#if (isinstance(data, BaseClass)):
#    #data.__class__ = eval(data.attrs[hdf5._CLASS_ATTR])
#    data = S.cast(data, MyClass2)

#print data._node.__class__.__module__ + "." + type(data._node).__name__

##type = eval (data._node.__class__.__module__ + "." + type(data._node).__name__)


print (type(data).__name__)