####################################

##      GRAPH BASED EXAMPLE       ##

####################################

# The idea is to develop a graph based workflow to work like Unreal Engine and Material Editor

import numpy as np
import json
import uuid

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
    return uuid.uuid4().get_hex()

# Following class will take the arguments and it will generate all the attributes.
# Also the base class contains a class to deserialize the class into a JSON scheme.
class BaseClass (object):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

#Nodes: Constant, Variable, functions (sum, substract, divide, et..), and mode complex nodes, layers, optimizers, etc..

# The way I must to think with Nodes is like a JSON schema. For example a simple Node like SUM will be the following.
# First declare the attributes or parameters for the Node. These parameters will be inserted into the dictionary of the base class.

# Possible Origins for parameters:

# 1. Parameter is a default value (1, 45.4, "Marco", etc..). Depending on the type of input or outputs this parameter sometimes has no sense.
# 2. Parameters is a default function. This mean we could implemente a custom lambda function that will correspond to the origin. This function will use
#    some of the attributes of the node Inputs, params, etc... For a parameter could be defficult to know each paramter of the function implemented to evaluate bt itself.
#    -> Could be an option that a parameter will be able to evaluate it self using the parameters the user ha configured.
# 3. Parameters is a link between other parameters. In this case the parameter must be evaluated recursively until found a final value 
# 4. For paramters that are neither Inputs or outputs these value could be anything.

# Note: think attrib value must be evaluated every time the parameter is called... not saved. Maybe eval must be called by value instead.. 

Nodes = {}

class BaseNode (BaseClass):
    def __init__(self, *args, **kwargs):
        if ("id" not in kwargs):
            kwargs["id"] = getID()
        return super(BaseNode, self).__init__(*args, **kwargs)
    def add(self, parameter):
        self.__dict__[parameter.name] = parameter
    def get (self, name):
        return self.__dict__[name]
    def eval(self):
        for key in self.__dict__:
            param = self.__dict__[key]
            if (isinstance(param,parameter)):
                param.eval()
 
class parameter (BaseClass):
    def __init__(self, *args, **kwargs):
        return super(parameter, self).__init__(*args, **kwargs)
    @staticmethod
    def create (nid, name, dtype = "float", slot = 0, category = 0, origin = None, value = 0):
        '''
        This class will create a new instance by using the arguments specified. 
        This function will ensure parameter is created with all neccesary mandatory fields.
        Parameters:
            "dclass": type of the current class to be cast. 
            "nid":  Host Node Id that will contain this parameter. The Unique key will be id + name
            "name": name of the attribute 
            "dtype": type of the parameter: "float", "integer", "np.ndarray", "string", "function", etc...
            "slot": integer
                0: Input
                1: Output
                2: Other
            "category": 0, 1, 2 for public, private, protected, etc.. to classify this parameter into a category
            "origin": This will indicate the origin of the value. Could be None, inferred instance from another parameter (node) or a lambda function
           
            "value": this is the current value of the parameter after it's evaluation (depending of the origin)
        '''
        json = {"dclass": type(parameter()), "nid": nid, "name": name, "dtype": dtype, "slot": slot,  "category": category, "origin": origin, "value": value}
        return parameter(**json) # Un-pack fields
    def eval(self):
        # Check wether the parameter is connected to another with a link between them
        if (isinstance(self.origin, parameter)):
            self.value = self.origin.value
        elif (self.dtype == "function"): # Check if the paramter it's a custom function to evaluate
             # Call to a global function to compile evaulate it
            self.value = self.evaluate_function(self.origin)
    def evaluate_function (self,strfunc):
        # Extract the parameters in the funcion in order to evaluate it
        params = strfunc.split(":")[0].replace ("lambda","", 1).strip().split(",")
        values = [Nodes[self.nid].get(param.strip()).value for param in params]
        return eval(strfunc)(*values)
    
class nsum (BaseNode):
    def __init__(self, *args, **kwargs):
        return super(nsum, self).__init__(*args, **kwargs)
    @staticmethod
    def create (x,y,o):
        json =  {"id": getID(),
                "x": x,
                 "y": y,
                 "output": output 
                  }
        return nsum(**json) # Un-pack fields
    def eval(self):
        super(nsum, self).eval()
        self.output.value = sum(self.x.value, self.y.value)

        
##############
#### TEST ####
##############

# Create the Custom Node
mynode_1 = BaseNode()

pInputX_1 = parameter.create(mynode_1.id,"x","float",0,0,None,50)
pInputY_1  = parameter.create(mynode_1.id,"y","float",0,0,None,50)
pOutput_1 =parameter.create(mynode_1.id,"output","function",1,0,"lambda x, y:  x + y",0)

mynode_1.add(pInputX_1)
mynode_1.add(pInputY_1)
mynode_1.add(pOutput_1)

mynode_2 = nsum()

pInputX_2 = parameter.create(mynode_2.id,"x","float",0,0,None,25)
pInputY_2  = parameter.create(mynode_2.id,"y","float",0,0,None,25)
pOutput_2 =parameter.create(mynode_2.id,"output","float",1,0, None,0)

mynode_2.add(pInputX_2)
mynode_2.add(pInputY_2)
mynode_2.add(pOutput_2)

mynode_3 = nsum()

pInputX_3 = parameter.create(mynode_3.id,"x","float",0,0,pOutput_2,0)
pInputY_3  = parameter.create(mynode_3.id,"y","float",0,0,pOutput_1,0)
pOutput_3 =parameter.create(mynode_3.id,"output","float",1,0, None,0)

mynode_3.add(pInputX_3)
mynode_3.add(pInputY_3)
mynode_3.add(pOutput_3)

# Add the node into the list
Nodes[mynode_1.id] = mynode_1
Nodes[mynode_2.id] = mynode_2
Nodes[mynode_3.id] = mynode_3

# eval the node whatever it does
for nodeid in Nodes:
    Nodes[nodeid].eval()
    print (Nodes[nodeid].__dict__)
    print (Nodes[nodeid].toJSON())

print ("END")


# In this case the eval() will only update the attributes, so this function no longer will return any value.
# Also the tool must be capable to link between the instances.

# Try to write an output file JSON format and deserialize into classes.
# Also the result might be the same

# Add mode functionality to the nodes and think in:
# 1. Declaring private parameters
# 2. How to use these parameters
# 3. How to implemente ppost-process or post-process
# 4. Assemblies, etc...

# X.  Implement all current configuration in Singularity with nodes.
# x2. Develop a Node.js, MongoDB and JavaScript web basec application.

# ... mode to come. The idea is to have something similar to Unreal in order to visualize the Network, results,e tc...
