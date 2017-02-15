import numpy as np
import theano
import sys
import gzip
from six.moves import cPickle


def serialize(data, protocol = 0):
  return cPickle.dumps(data, protocol)

def deserialize(data):
  return cPickle.loads(data)

name = "This is a string"
i = 100
v = [0,1,2,3,4,5]
d = {"name": "Javier", "lastname":"Santos Andr�s", "age":33}
t = (4,None, 5)
#t = (4,None, 5)

ser = serialize(t)
print (ser)
print (type(ser).__name__)

complexn = "_S%" + ser + "%S_"
if (complexn.startswith("_S%") and  complexn.endswith("%S_")):
    complexn = complexn[3:-3]

if (complexn == ser):
    deser = deserialize(complexn)


deser = deserialize(ser)
print (deser)
print (type(deser).__name__)


#######@######
# Base Class #
########@##### 

class AnotherClass (object):
    def __init__(self, name):
        self._name = name

# The main pourpose with this base class is to generalize the  process of manipulation of all main attributes and parameters

class BaseClass (object):
    def __init__(self, **kwargs):
        self._attribs = {} 
        for key in kwargs:
            self._attribs[key] = kwargs[key]
            setattr(self, key, self._attribs[key]) #This will add a function called key that returns the value given by parameters
    @property
    def attrs(self):
        return self._attribs   
    def attribs2(self, atrib):
        return getattr(self,atrib)   # This gets the atrib that was previously set by setatts. Another way is by directly get the key from self._attribs variable.


#########
# MyClass #
######### 

class MyClass (BaseClass):
    def __init__(self, inputs = None, outputs  = None, loss = None, optimizer = None, regularizers = [], name = None):
        # This code will print all the values from paramters passed by the function automatically
        # Also will return self -> This could be useful to know the instance that was instanced prior to the super classes
        parameters = locals().keys()
        for key in parameters:
            #Do something with the parameters before the will be passes through the super class
            print "key:{0} = {1}".format(key, locals().get(key))

        self._inputs = self._get_nodes(inputs)
        self._outputs = outputs
        self._loss = loss
        self._optimizer = optimizer
        self._regularizers = regularizers
        self._name = name
        self._compiled = False
        self._predict = None
        self._train = None

        # Also this could be done manually. name is passed by parameters as literal, so if name = "Javier", this would pass "Javier" = "Javier". 
        # This could be useful if the class need to do previous computation with the parameters.

        kwargs = {"inputs":self._get_nodes(inputs), "outputs":outputs, "name": name}
        super(MyClass, self).__init__(**kwargs)

    def _get_nodes(self, nodes):
        if (nodes == None):
            result = []
        elif (isinstance(nodes,tuple)):
            result = []
            result.append(nodes)
        else:
            result = nodes
        return result 

def is_collection(x):
    if (isinstance(x, (list, tuple, dict, np.ndarray))):
        return True
    else:
        return False

#def is_primitive(data):
#    if (is_collection(data)):
#        if (isinstance(data,dict)):
#            data = data.values()
#        for item in data:
#            if not(isinstance(item, (int, str, float, type, bool))):
#                return False
#        return True
#    elif (isinstance(data, (int, str, float, type, bool))):
#        return True
#    else:
#        return False

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
    
def is_none_or_empty(x):
    if (x is None):
        return True
    elif is_collection(x) and not len(x):
        return True
    else:
        return False

#test = BaseClass(name = "Javier", inputs = [(23,23), (None, 65)], outputs = (None, 10))

#print ("CASE 1")
#print test.attribs["name"]
#print test.attribs["inputs"]
#print test.attribs["inputs"][0]
#print test.attribs["inputs"][1]
#print test.attribs["outputs"]

#print ("CASE 2")
#print test.name
#print test.inputs
#print test.inputs[0]
#print test.inputs[1]
#print test.outputs

#print ("CASE 3")
#print test.attribs2("name")
#print test.attribs2("inputs")
#print test.attribs2("outputs")

#test = MyClass(name = "Javier", inputs = [(23,23), (None, 65)], outputs = (None, 10)) # Error in hdf5
test = MyClass(name = "Javier", inputs = [(None,23),(None,65)], outputs = (5, 10))
print test.name
#Can I change the paramters?
test.name = "Pedro"
print "Now I'm", test.name

test2 = MyClass(name = "Rodrigo", inputs = [(23,23), (None, 65)], outputs = (None, 10))
print test2.name
# Attributes are declared statics?
test2.name = "Marta"
print "Now I'm", test2.name
print "But I'm still calling me", test.name


#What can I do to reconstruct this structure from the attributes and class

# I must save the original attributes passed by the functions with the constructor....
# And finally all the settings that could have been changed during he execution.

# HDF5 -> Reconstruct the object but using the classes implmented with the arguments stored and attributes.
# List objects

name = "This is a string"
i = 100
v = [0,1,2,3,4,5]
d = {"name": "Javier", "lastname":"Santos Andr�s", "age":33}
t = (4,None, 5)
#t = (4,None, 5)

def sum (x,y):
    return x + y

func = sum

n = np.asarray(range(5), dtype = theano.config.floatX)
print (type(n).__name__)

p = np.asarray(range(3*4*2), dtype = 'int32').reshape(3,4,2)

node1 = {"name": "node1", "val":v, "param": n}
node2 = {"name": "node2", "val":t, "param": p}
nodes = {"node1":node1, "node2":node2}

m = {"name": "model", "nodes":nodes, "test":test, "another":AnotherClass("pedro"), "myfunc":func}

print m

print (is_complex(n))

print (type(n).__name__)

print (is_complex(name))
print (is_complex(i))
print (is_complex(v))
print (is_complex(d))
print (is_complex(t))
print (is_complex(node1))
print (is_complex(node2))
print (is_complex(nodes))

print (is_complex(test))
print (is_complex(test2))


print (is_complex(func))
print (type(func).__name__)





for item in m:
    print item
    
#The idea is to reconstruct the same hierarchy as m for the class for all the attributes.
# Initial attributes will have the setattr to define basic functionality.

# Tee best idea is to reconstruct a hierrchy of this type is to do a recursive function that iterate through all the elements recursively checking the type for each input
# and genearting the entire structure- > This will mean the save hdf5 is going to itarate throgh all the elements and create the structure

print "+++++++++++++++++++++++"
print " START FROM THIS LINE"
print "+++++++++++++++++++++++"

## Loop over all the strucuture (not the hdf5 file)
#def structure (name, data, hdf5):
#    if (not is_none_or_empty(data)):
#        if (is_collection(data) and not is_primitive(data)):
#            #Create a new group
#            print ("Group {}".format(name))
#            if (isinstance(data, (tuple, list))):
#                # Data is wether a tuple or a list
#                for index, item in enumerate(data):
#                    #print "index[{0}] = {1}".format(index, item)
#                    # If the variable is a 
#                    structure(str(index), item, hdf5)
#            else:
#                #data is a dictionary. Loop over if 
#                for key, item in data.iteritems():
#                    #print "key[{0}] = {1}".format(key, item)
#                    # If the variable is a 
#                    structure(key, item, hdf5)
#        else:
#            #Create a single attribute or data set
#            if (isinstance(data,BaseClass)):
#                structure(name, data.attrs,hdf5)
#            elif (is_primitive(data)):
#                print "key[{0}] = {1}".format(name, data)
#            else:
#                print "key[{0}] = {1}".format(name, type(data).__name__)
                


# Loop over all the strucuture (not the hdf5 file)
def structure (name, data, hdf5):
    if (not is_none_or_empty(data)):
        if (is_collection(data) and not is_primitive(data)):
            #Create a new group
            hdf5 += "   "
            print (hdf5 + "Group {}".format(name))
            if (isinstance(data, (tuple, list))):
                # Data is wether a tuple or a list
                for index, item in enumerate(data):
                    #print "index[{0}] = {1}".format(index, item)
                    # If the variable is a 
                    structure(str(index), item, hdf5)
            else:
                #data is a dictionary. Loop over if 
                for key, item in data.iteritems():
                    #print "key[{0}] = {1}".format(key, item)
                    # If the variable is a 
                    structure(key, item, hdf5)
        else:
            #Create a single attribute or data set
            if (isinstance(data,BaseClass)):
                structure(name, data.attrs,hdf5 + "   ")
            elif (is_primitive(data)):
                print hdf5 + "key[{0}] = {1}".format(name, data)
            else:
                print hdf5 + "key[{0}] = {1}".format(name, type(data).__name__)
                


structure("rook",m, "")

import h5py
import datetime


def get_deserialized_data(data):
    if (is_serialized(data)):
        return deserialize(data[3:-3])
    else:
        return data

def get_serialized_data(data):
    return "_S%" + serialize(data) + "%S_"

def is_serialized(data):
    if (isinstance(data, str) and data.startswith("_S%") and data.endswith("%S_")):
        return True
    else:
        return False


testFuncSer = get_serialized_data(func)


# Save model. The idea is  to give a initial data that can be a list or a class that inhertance from BaseClass with a list of attributes and serialize the data into a HDF% format
#The idea is to be able to reconstruct this data again by creating an instance directly with eval function or using a load function to load the settings loaded into a BaseClass.

# definition to differentiate from collections using indexes 
_TYPE_ATTR = "type"

def save_settings(filename, data, rootname = "settings"):
    # Recursive function that iterate over all the elements that will be saved into the file
    def save(name, data, hdf5):
        # Check there is data to store into an hdf file
        if (not is_none_or_empty(name)):
            # Check wether is a collection or a primitive data inside a collection, like np, int, string, etc... 
            if (is_collection(data) and is_complex(data)):
                #Create a new group for the collection
                group = hdf5.create_group(name)
                group.attrs[_TYPE_ATTR] = type(data).__name__
                # Check if the collection is a tuple or a list to loop over the items with an iterator.
                if (isinstance(data, (tuple, list))):
                    # Data is wether a tuple or a list
                    for index, item in enumerate(data):
                        # Do recursive call using the current prefix and index for each item
                        save(str(index), item, group)
                else:
                    #data is a dictionary. Loop over it and use the same key to store the data
                    for key, item in data.iteritems():
                        # Do recursive call using the current key for each item
                        save(key, item, group)
            else:
                #Create a single attribute or data set
                if (isinstance(data,BaseClass)):
                    ## Because is a base class we can iterate through all the attributes stored in the base class
                    save(name, data.attrs,hdf5)
                elif (not is_complex(data)):
                    if (is_collection(data)):
                        try:
                            hdf5.create_dataset(name, data=data, compression="gzip", compression_opts=9)
                        except:
                            hdf5.attrs[name] = get_serialized_data(data)
                    else:
                        hdf5.attrs[name] = data
                else:
                    try:
                        # If the object is something with unknow type then we are going to try the serialization using pickl dump
                        hdf5.attrs[name] = get_serialized_data(data)
                    except:
                        # If fails then we store the name of the type in order to store somthing about it
                        hdf5.attrs[name] = type(data).__name__

    # Start the process of saveing the settings
    file = h5py.File(filename, "w")
    save(rootname,data, file)
    file.close()  
    

save_settings("test.hdf", m, "root")

print "+++++++++++++++++++++++"
print " START FROM THIS LINE"
print "+++++++++++++++++++++++"

def load_settings(f, spaces):
    for key in f.attrs:
        attr = f.attrs[key]
        if (is_serialized(attr)):
            attr = get_deserialized_data(attr)
        string = spaces + key + ": {0}" 
        print string.format(attr)
        #print (spaces + attr + ":" + str(attr))
   
    for key in f.keys():
        item = f.get(key)
        if (isinstance(item, h5py.Dataset)):
            data = np.array(item)
            string = spaces + key + ": {0}" 
            print string.format(data)
            #print (spaces + key + ":" + str(data.shape))
        else:
            print (spaces + key )
            load_settings(item,spaces + "  ")

f = h5py.File("test.hdf")
load_settings(f,"")
f.close()


# This will generate the data accordingly the data stored in the hdf5 file
# Now the idea is to load the data and try to generate the same strucuture as the original... with the same classes and data.

def load_settings(filename):
    
    def set_data(data, key, item):
        if (isinstance(data, (list))):
            data.append(item)
        elif (isinstance(data, (tuple))):
            data = list(data)
            data.append(item)
            data = tuple(data)
        elif (isinstance(data,dict)):
            data[key] = item
        elif (isinstance(data,BaseClass)):
            data.attrs[key] = item   
        else:
            data = item
        return data
    
    def load(hdf5):
        # Get the attribute type in order to get the class wher the data will be stored
        foo = None
        type = None
        if _TYPE_ATTR in hdf5.attrs:
            type = hdf5.attrs[_TYPE_ATTR]
            #Create the object
            foo = eval(type + "()")
            # del the type key from the attributes list
            del hdf5.attrs[_TYPE_ATTR]

        # Start adding data to the object
        for key in hdf5.attrs:
            attr = hdf5.attrs[key]
            if (is_serialized(attr)):
                attr = get_deserialized_data(attr)
            foo = set_data(foo,key,attr)
    
        # Creates the gruop elements and operates in the same way as attributes
        for key in hdf5.keys():
            item = hdf5.get(key)
            if (isinstance(item, h5py.Dataset)):
                foo = set_data(foo,key,np.array(item))
            else:
                foo = set_data(foo,key,load(item))
 
        return foo

    # Start the parsing of the file
    file = h5py.File(filename)
    data = load(file)
    file.close()

    return data

# Load the data settings
data = load_settings("test.hdf")

print (type(data).__name__)