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
        result = {}
        for key in self.__dict__:
            param = self.__dict__[key]
            if (isinstance(param,parameter)):
                result[key] = param.eval()
        return result
 
class parameter (BaseClass):
    def __init__(self, *args, **kwargs):
        return super(parameter, self).__init__(*args, **kwargs)
    @staticmethod
    def create (nid, name, dtype = "float", slot = 0, category = 0, origin = None):
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
            "origin": This will indicate the origin of the value. Could be a value, inferred instance from another parameter or a lambda function
        '''
        json = {"dclass": type(parameter()), "nid": nid, "name": name, "dtype": dtype, "slot": slot,  "category": category, "origin": origin}
        return parameter(**json) # Un-pack fields
    def eval(self):
        # Check wether the parameter is connected to another with a link between them
        if (isinstance(self.origin, parameter)):
            return self.origin.eval()
        elif (self.dtype == "function"): # Check if the paramter it's a custom function to evaluate
             # Call to a global function to compile evaulate it
            return self.evaluate_function(self.origin)
        else:
            # Return the default orig 
            return self.origin
    def evaluate_function (self,strfunc):
        # Extract the parameters in the funcion in order to evaluate it
        params = strfunc.split(":")[0].replace ("lambda","", 1).strip().split(",")
        values = [Nodes[self.nid].get(param.strip()).eval() for param in params]
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
        return {"output", sum(self.x.eval(), self.y.eval())}       #What will happen if the parameter is inferred instead.

##############
#### TEST ####
##############

## Create the Custom Node
#mynode_1 = BaseNode()
#mynode_2 = nsum()

#pInputX_1 = parameter.create(mynode_1.id,"x","float",0,0,19)
#pInputY_1  = parameter.create(mynode_1.id,"y","float",0,0,23)
#pOutput_1 =parameter.create(mynode_1.id,"output","function",1,0,"lambda x, y:  x + y")
#pOutput_2 =parameter.create(mynode_1.id,"output","float",1,0, None)

#mynode_1.add(pInputX_1)
#mynode_1.add(pInputY_1)
#mynode_1.add(pOutput_1)

#mynode_2.add(pInputX_1)
#mynode_2.add(pInputY_1)
#mynode_2.add(pOutput_2)

## Add the node into the list
#Nodes[mynode_1.id] = mynode_1
#Nodes[mynode_2.id] = mynode_2

## eval the node whatever it does
#print (mynode_1.eval())
#print (mynode_1.output)

#print (mynode_2.eval())
#print (mynode_2.output)

# This Test execute succesfully a custom node and a predefined node.
# THe next step is to be able of create a connection between the nodes. inputs-output
