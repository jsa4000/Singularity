##########################
# TEST NODE BASED MODULE #
##########################

# The idea with a node based workflow is to simplify the generation of any logic or bahaviour without the need to generate python code.
# This module is to generate the basic functionality to dynamically create a workflow model based on nodes using JSON for definning all the behaviour and parameters.

# The approach is going to be based in Nodes so a node can be connected to multiple nodes.
# Each node could have multiple inputs (properties) and outputs (functions). The atrributes (channels) can be linked with the outputs.
# The nodes process an evaluation function. 

# An ideas is that Nodes will inherit from an class that also perform aome task to ppre-process and post-porcess the data after and before the evaluation takes place.
# Some links regarind this implementation are:
#   http://gamedev.stackexchange.com/questions/88199/implementing-a-graph-based-material-system
#   http://stackoverflow.com/questions/645354/implementing-a-node-based-graphical-interface
#   http://stackoverflow.com/questions/5493474/graph-implementation-c
#   http://wiki.tcl.tk/8537

# In order to draw the nodes in order to compose the Flowchart the best option in to consider the JavaScript library for this pourpose.
# http://www.debasish.in/2014/02/building-assembly-control-flow-graphcfg.html

#BASIC COMPONENTS

    # Nodes will be composed by:
    
        # Attributes: These can be generic attributes like visible, locked, forward, backward, etc..)
        #             Also the attributes can be something specific to that Node, like temp variabbles, params, biases, files I/O, etc..
        #             This mean nodes not only will be composed by inputs amd outputs

        # Inputs: These inputs are like the initializations parameters that the node will use. Could be similar that constructors parameters.
        # Outputs: The output generically speaking will be unique. the output will be the result operation of the eval function.
        #           Also, the Outputs could be any attributes or some specific function implemented specific for that node. e.g. Get_Amount_Values, Get_Shapes.  

        # Inner funcitonality (Pre-process, post-process, evaluation, get_shape, etc...) This will be manage by an abstract class
        #               
    
        # NOTE: The implementation that will be used for the pre-process, evaluation, etc.. could be also generated using node based graph or python code insted. 
        #       Another approach is to indicate if the field is an input, output, inferred, ec

    #  Links and connections must be represented. 

        #The link are connection between a node to another. The links it will connect an outputs attribute from a Node to an input node from another.

        # NOTE: These Links are going to connect instances for Nodes from inputs to outputs. Instances are imprtant because each instance
        # will have theirs's own properties and configuration (initialization). 

    # Functions

        # Fnctions that will be used as interface to do all the operation independent from the technology, backend, etc being used

    # NOTES:
    #   However the implementation will be the same and the logic. For this particular reason it will
    #   be important to optimize the size of the data stored to serialize the instances in order to not repeat too much information. 
    #   In HDF5 ard cpickle modulues the serialization is done by using the internal binary codification. This mean the loginc will be stored for each instance. I must differenciate
    #   between instance and new assembly or configuration.
 
# In order to do that we need to create a basic classes with this functionality before doinf anythin

# Base Class
# Base class will have the basic strucuture to create a class from it's attributes.
# This class will have an sctrucuture similar to JSOn using dictionary in order to create the hierarchy and functionality to the classes.

#################
## PSEUDO-CODE code ##
#################

# Basic Class is intended to replicate the structure of a class from JSON structure.
# {}

# BASIC NODE
    # ATTRIBUTES

# LINK: 
    # FROM: OUTPUT_NODE_ID, OUTPUT_ATTRIBUTE    
    # TO: INPUT_NODE_ID, INPUT_ATTRIBUTE

class node(object):
    pass


# NOTE: Anothe thing is that some functions my not be a node... but at the end must be treated as a node operation or something. 
 

















# EXAMPLES TO CREATE DINAMICALLY OBJECT FROM A TYPE BASE FUNCTION.
   
#An example that craeated a class that inherits from the baseclass

#class BaseClass(object):
#    def __init__(self, classtype):
#        self._type = classtype

## This class returns a class that has been dinamically created 
#def ClassFactory(name, argnames, BaseClass=BaseClass):
#    def __init__(self, **kwargs):
#        for key, value in kwargs.items():
#            # here, the argnames variable is the one passed to the
#            # ClassFactory call
#            if key not in argnames:
#                raise TypeError("Argument %s not valid for %s" 
#                    % (key, self.__class__.__name__))
#            setattr(self, key, value)
#        BaseClass.__init__(self, name[:-len("Class")])
    
#    newclass = type(name, (BaseClass,),{"__init__": __init__})
#    return newclass

#def set_x(self, value):
#    self.x = value

## This add to the Subclass a new method that is already defined.
#SubClass = type('SubClass', (BaseClass,), {'set_x': set_x})
## (More methods can be put in SubClass, including __init__().)

##Example using this class
#obj = SubClass()
#obj.set_x(42)
#print obj.x  # Prints 42
#print isinstance(obj, BaseClass)  # True

##############################################

## EXAMPLE OF CREATING A DYNAMIC FUNCTION ##

###############################################

#exec vs eval

#Create a function normally
def sum (x,y):
    return x + y

total = sum (3,2)
print (total)

# Create a function using Eval
sum_code = ["def sum2 (x,y):","    return x + y"]
for line in sum_code:
    eval (line)

# I supose the function has been already created
total2 = sum2 (3,2)
print (total2)

# ==> ERROR <==
#  eval is used to evaluate function like "1 + 1" or x + 1, etc..
# Used instead exec

# Two Options

# 1. LINE
# Create a function using exec
sum_code = ["def sum2 (x,y):\n  return x + y"]
for line in sum_code:
    exec (line)

# I supose the function has been already created
total2 = sum2 (3,2)
print (total2)

# 2. ARRAY
# Create a function using exec
arraycode = ["def sum3 (x,y):","    return x + y"]
code = ""
for index, line in enumerate(arraycode):
   code += line + ("\n" if index < len(arraycode) - 1 else "")

print (code)
exec (code)

# I supose the function has been already created
total3 = sum3 (3,2)
print (total3)

# Gereratin lambda function from the eval is also possibleknowing the parameters  involve in the function
#func_obj = lambda a, b: eval('a + b')

# The thing is that I must provide a Node based workflow but the old one must be work as usual. that means the Constant node.

# In order to be able to enconde / decode from JSON to an obejct I must use the following code:
# https://developer.rackspace.com/blog/python-magic-and-remote-apis/

# Lets see some of the functionality that are already built-in in python.

class Computer(CServOBJ):
    def __init__(self):
        self._attributes = None
        super(CServOBJ, self).__init__()

        @property
        def attribs(self):
            # Check If there is an attribue already defined in the class
            if not self._attribs:
                self._attribs = []
            return self._attribs

        def __dir__(self):
            return sorted(dir(type(self)) + list(self.__dict__) + self.attribs)

####################################

##       JSON EXAMPLE            ##

####################################

# This class is to generate a JSON code from an objecto.
# It's lile the tostring function (str) natively implemented in python but the use is to generte a JSON from the class.

import json

# Base class to serialize a class to JSON
class Object:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    def fromJSON(self, json):
        self.__dict__ = json


# Base class to deserialize a class from JSON

from datetime import datetime

def getID():
    return datetime.strftime(datetime.now(), '%Y%m%d%H%M%S%f')

class Seat(Object):
    def __init__(self, number, type):
        self._instanceID = getID()
        self._number = number
        self._type = type

class Wheels(Object):
    def __init__(self, number, type):
        self._instanceID = getID()
        self._number = number
        self._type = type
        self._pressure = 100
        self._status = False
    def check(self):
        if (self._pressure <= 0):
            self._status = False
        else:
            self._status = True
        return self._status
    def eval (self, inputs):
        self._pressure -= inputs     
        return self.check()

class Vehicle (Object):
    def __init__(self, brand = None, model = None, color = None, number_of_wheels = None):
        self._instanceID = getID()
        self._brand = brand
        self._model = model 
        self._color = color
        self._wheels = Wheels(4, "Bridgestone")
        self._seats = [Seat(2, "drivers"), Seat(3, "Passengers"), Seat(1, "Baby")]
        self._status = False
        self._km = 0
    @property
    def km(self):
        return self._km
    def check(self):
        if (self._wheels.check()):
            self._status = False
        else:
            self._status = True
        return self._status
    def eval (self, inputs):
        self._km += inputs
        self._wheels.eval(inputs)
        return self.check()
    
myvehicle = Vehicle ("Seat", "Leon", "Gris", 4)

# the thing is the wheels can only 
for i in range (100):
    myvehicle.eval(10)
    if (not myvehicle.check()):
        print ("The Vehicle is broken: " + str(myvehicle.km * 10) + " km")
        break
   
serialize = myvehicle.toJSON()
print (serialize)

deVehicle = Vehicle()
deVehicle.fromJSON(json.loads(serialize))

# This sample show the limitation that every object created must be parsed and create from this dictionary recursively
print (deVehicle._brand)
print (deVehicle.check())

# An approach to this is to know the class of the object (property class or type), in my case since I know all the nodes will inherit from a base node, the will share the same 
# strucuture it will be easy to cast the dictionary into the particular NodeBase

print ("done!")

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

NodesCollecion = {}

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
        if (isinstance(self.orig, parameter)):
            return self.orig.eval()
        elif (self.dtype == "function"): # Check if the paramter it's a custom function to evaluate
             # Call to a global function to compile evaulate it
            return _evaluate_function(self.orig)
        else:
            # Return the default orig 
            return self.orig
    def _evaluate_function (strfunc):
        # Extract the parameters in the funcion in order to evaluate it
        params = strfunc.split(":")[0].replace ("lambda","", 1).strip().split(",")
        values = [nodes[NodesCollecion][param] for param in params]
        
        # return eval(self.orig)(get_node_parameters(self.orig)) #Call to a global function to evaulate it
        return eval(strfunc)(*values)

# The next thing is to build the node. 

# Nodes are going to have a list of parameters. This list of parameter are going to be used for the interval process of the node in order to bring some functionality.

# For each operation some functionality must be defined. This definition could be implemented in serveral way.
# 1. By coding the functionality directly into the source code: def eval(): output = sum(x,y)
# 2. By directly set the output parameters value by default or defining a code so it will be evaluated (executed) in real-time.
# 3. By defining a workflow to compute this operation

# NOTE: I think the graph model must provide all the possible method


# Case 1: By coding the functionality into a Node

# This means the calls to a functions will be directly made by using code and the inputs and outputs set into the node. This means the node are going to have the values
# set at the createion. However this values and connections must be saved and serialized/deserialized into JSON.  

jsonnode = {"id": getID(),
            "x": pInputX,
            "y": pInputY,
            "o": pOutput 
            }
    
class BaseNode (BaseClass):
    def __init__(self, *args, **kwargs):
        return super(BaseNode, self).__init__(*args, **kwargs)
    def eval(self):
        pass

class nsum (BaseNode):
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
        if (self.o.dtype == "function"):
            self.o.value = self.o.eval(self.x.value, self.y.value)
        else:
            self.o.value = sum(self.x.value, self.y.value)       #What will happen if the parameter is inferred instead.
        return True     # Chack i all parameters are correctly initialized



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

# Case 2: By directly set the output parameters value by default or defining a code so it will be evaluated (executed) in real-time.
# In order to do that the node must specify the operation to do inside the definition of the node... or in the definition of each output param, since the node could have
# different outputs.

#If the parameter it's a function the the node will use the output as a function instead get it's value


# In this case the inferred values that paramters could also have are not taking into accoun this time. For this reason is important to get the val outputs by previously do an
# evaluation for all nodes in the graph previously. This can be done id there is an change.

jsonInputX_2 = {"nid" : "201609112334455556",      # Node Id that host this parameter. The Unique key will be id + name 
             "name": "x",                       # name of the attribute ( This could be obtained for the key where will be contained...)
             "dclass": "parameter",             # class of the current dict to be cast
             "dtype": "float",                  # type of the parameter, float, integer, string, etc..
             "port": 0,                         # 0, 1, 2 for Input, Output, Param, etc..
             "cat": 0,                          # 0, 1, 2 for public, private, protected, etc.. or could be some custom category for separation 
             "val": 19                     # Current Value of the parameter (or default). Could be a value or a inferred instance from another parameter
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
              "val": "lambda x, y:  x + y"                       # This paramters could be a list, value, link(s) to another parameter, etc..
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

#NOTE: Add an addiional attribute in paramters with the function and val is used to set the value instead.
# Added the eval property to the parameters in case the type will be a function instead. in this case the 








# Case 3: By defininig a Base node workflow to perform this operation. 
# In order to do that prior to this it's needed to define default nodes with some functionality.

# The assembly is a group based on nodes, parameters and link that will have several ouptus and generate some inputs. Basically is the same conception as the nodes
# but the functionality is to allow more complex operations.



