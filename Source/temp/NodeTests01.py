
####################################

##      GRAPH BASED EXAMPLE       ##

####################################

# The idea is to develop a graph based workflow to work like Unreal Engine and Material Editor

import numpy as np
import json

# First, we need some implementation about functions and inti

def sum (x,y):
    return x + y

def substract (x,y):
    return x - y

def multiply (x,y):
    return x * y

def divide (x,y):
    return x / y

def sqrt (x):
    return x**2

def zeros (shape, type = np.float):
    return np.zeros(shape,dtype = type)

def random2D (shape, type = np.float):
    return np.random.rand(shape[0],shape[1])

miarray = random2D((2,3))
print (miarray)


# Second, I need a basic node Which I will use as a base for the creation. Serialization and deserialization will be also implemented

from datetime import datetime

# This instance will be used to create the instances
def getID():
    return datetime.strftime(datetime.now(), '%Y%m%d%H%M%S%f')

# Following class will take the arguments and it will generate all the attributes.
# Also the base class contains a class to deserialize the class into a JSON scheme.
class BaseClass (object):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

#Nodes: Constant, Variable, functions (sum, substract, divide, et..), and mode complex nodes, layers, optimizers, etc..

#The way I must to think with Nodes is like a JSON schema. For example a simple Node like SUM will be the following.
# First declare the attributes or parameters for the Node. These parameters will be inserted into the dictionary of the base class.
       
class parameter (BaseClass):
    def __init__(self, *args, **kwargs):
        return super(parameter, self).__init__(*args, **kwargs)
    @property
    def value (self):
        if (isinstance(self.val,parameter)):
            return self.val.value
        else:
            return self.val
    @value.setter
    def value(self, value):
        self.val = value
    @staticmethod
    def create (nid, name, dtype = "float", port = 0, cat = 0, orig = None, val = None):
        '''
             "dclass": class of the current dict to be cast
             "nid":  Node Id that host this parameter. The Unique key will be id + name
             "name": name of the attribute ( This could be obtained for the key where will be contained...)
             "dtype": type of the parameter, float, integer, string, etc..
             "port": 0, 1, 2 for Input, Output, Param, etc..
             "cat": 0, 1, 2 for public, private, protected, etc.. or could be some custom category for separation 
             "orig": This will indicate the origin of the value. Could be a value, inferred instance from another parameter or a lambda function
             "val":  Current Value of the parameter (or default). 

        '''
        json = {"dclass": parameter.__name__,
                "nid": nid,
                "name": name,
                "dtype": dtype,
                "port": port,
                "cat": cat,
                "orig": orig,
                "val": val
             }
        return parameter(**json) # Un-pack fields


# Declare the parameters as json isntance. This structure will be similar as the data that will come from a database or a file. (With default values, etc..)

jsonInputX = {"nid" : "201609112334455555",      # Node Id that host this parameter. The Unique key will be id + name 
             "name": "x",                       # name of the attribute ( This could be obtained for the key where will be contained...)
             "dclass": "parameter",             # class of the current dict to be cast
             "dtype": "float",                  # type of the parameter, float, integer, string, etc..
             "port": 0,                         # 0, 1, 2 for Input, Output, Param, etc..
             "cat": 0,                          # 0, 1, 2 for public, private, protected, etc.. or could be some custom category for separation 
             "val": 19                          # Current Value of the parameter (or default). Could be a value or a inferred instance from another parameter
            }

jsonInputY = {"nid" : "201609112334455555",
              "name": "y",
              "dclass": "parameter",
              "dtype": "float",
              "port": 0,
              "cat": 0,
              "val": 23
             }

jsonOutput = {"nodeid" : "201609112334455555",
              "name": "output",
              "dclass": "parameter",
              "dtype": "float",
              "port": 1,                        # Since it's an outputs the flag is 1
              "cat": 0,
              "val": None                       # This paramters could be a list, value, link(s) to another parameter, etc..
             }

pInputX = parameter (**jsonInputX)      # Unpack dictionary fields
pInputY  = parameter (**jsonInputY)
pOutput = parameter (**jsonOutput)

pParamTemp = parameter.create("201609112334455555","temp","string",2,0,0)

# Now all the parameters are correctly created

print pInputX.toJSON()
print pInputY.toJSON()
print pOutput.toJSON()
print pParamTemp.toJSON()

# Also the tool must deal with the link between the nodes. This can be another class or inferred from the val key of the paramters

class link (BaseClass):
    def __init__(self, *args, **kwargs):
        return super(link, self).__init__(*args, **kwargs)
    @staticmethod
    def create (pfrom,pto):
        json = {"id": getID(),
                "From" : pOutput,
                "To": pInputX
             }
        return link(**json) # Un-pack fields

#The link will be stored as follows:

jsonLink = {"id": getID(),
            "From" : pOutput,
            "To": pInputX
             }

link = link(**jsonLink)
print link.toJSON()


# The next thing is to build the node

# Nodes are going to have a list of parameters. This list of parameter are going to be used for the interval process of the node in order to bring some functionality.

# For each operation some functionality must be defined. This definition could be implemented in serveral way.
# 1. By coding the functionality directly into the source code: def eval(): output = sum(x,y)
# 2. By defining a workflow to compute this operation
# 3. By directly set the output parameters value by default 
# NOTE: I think the graph model must provide all the possible method


# 1. Case 1: By coding the functionality
# This means the calls to a functions will be directly made by using code and the inputs and outputs set into the node. This means the node are going to have the values
# set at the createion. However this values and connections must be saved and serialized/deserialized into JSON.  

jsonnode = {"id": getID(),
            "x": pInputX,
            "y": pInputY,
            "o": pOutput 
            }
    
class nsum (BaseClass):
    def __init__(self, *args, **kwargs):
        return super(nsum, self).__init__(*args, **kwargs)
    @staticmethod
    def create (x,y,o):
        json =  {"id": getID(),
                "x": x,
                 "y": y,
                 "o": o 
                  }
        return nsum(**json) # Un-pack fields
    def eval(self):
        self.o.value = sum( self.x.value, self.y.value)       #What will happen if the parameter is inferred instead.
        return True     # Chack i all parameters are correctly initialized

nodeSum = nsum(**jsonnode)

# eval the node whatever it does
nodeSum.eval()
print (nodeSum.o.value)

# In this case the inferred values that paramters could also have are not taking into accoun this time. For this reason is important to get the val outputs by previously do an
# evaluation for all nodes in the graph previously. This can be done id there is an change.

jsonInputX_2 = {"nid" : "201609112334455556",      # Node Id that host this parameter. The Unique key will be id + name 
             "name": "x",                       # name of the attribute ( This could be obtained for the key where will be contained...)
             "dclass": "parameter",             # class of the current dict to be cast
             "dtype": "float",                  # type of the parameter, float, integer, string, etc..
             "port": 0,                         # 0, 1, 2 for Input, Output, Param, etc..
             "cat": 0,                          # 0, 1, 2 for public, private, protected, etc.. or could be some custom category for separation 
             "val": pOutput                     # Current Value of the parameter (or default). Could be a value or a inferred instance from another parameter
            }

jsonInputY_2 = {"nid" : "201609112334455556",
              "name": "y",
              "dclass": "parameter",
              "dtype": "float",
              "port": 0,
              "cat": 0,
              "val": 23
             }

jsonOutput_2 = {"nodeid" : "201609112334455556",
              "name": "output",
              "dclass": "parameter",
              "dtype": "float",
              "port": 1,                        # Since it's an outputs the flag is 1
              "cat": 0,
              "val": None                       # This paramters could be a list, value, link(s) to another parameter, etc..
             }

pInputX_2 = parameter (**jsonInputX_2)      # Unpack dictionary fields
pInputY_2  = parameter (**jsonInputY_2)
pOutput_2 = parameter (**jsonOutput_2)

jsonnode_2 = {"id": getID(),
            "x": pInputX_2,
            "y": pInputY_2,
            "o": pOutput_2 
            }
    
nodeSum_2 = nsum(**jsonnode_2)

# eval the node whatever it does
nodeSum_2.eval()
print (nodeSum_2.o.value)
