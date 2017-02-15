####################################
####################################
##      GRAPH BASED EXAMPLE       ##
####################################
####################################

import numpy as np
import json
import uuid

# The idea is to declare pre-defined structures like parameters, nodes, etc to be used with different nodes before creating instances. 
# Each instances will generate ids to differs between them.

# Configuration are the default Nodes this module provides.
# Node:          [id, name]
# Parameters:    [id, nodeId, name, dtype, slot, category, originId, value]
        
# Dictionay with the info of the instances
GLOBAL_PARAMETERS = {}
GLOBAL_NODES = {}

NODE_INFO =  ["id", "name"]
PARAMETER_INFO =  ["id", "nodeId", "name", "dtype", "slot", "category", "originId", "value"]

def sum (x,y):
    return x + y

def substract (x,y):
    return x - y

# This instance will be used to create the instances
def generateUId():
    return uuid.uuid4().get_hex()

class BaseClass (object):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
    def deserialize(self):
        return [value for (key, value) in self.__dict__.items() ]

class Parameter (BaseClass):
    def __init__(self, *args, **kwargs):
        kwargs.update ({PARAMETER_INFO[index] : value  for index, value in enumerate(args)})
        super(Parameter, self).__init__(*args, **kwargs)
    def eval(self):
        if (self.originId is not None):
            if (self.dtype == "function"):
                self.value = self.eval_function(self.originId)
            else:
                origin = GLOBAL_PARAMETERS[self.originId]
                origin.eval()
                self.value = origin.value
    def eval_function (self,strfunc):
        params = strfunc.split(":")[0].replace ("lambda","", 1).strip().split(",")
        inputs = [GLOBAL_PARAMETERS[GLOBAL_NODES[self.nodeId].parameters[param.strip()]] for param in params]
        map(lambda input: input.eval(), inputs) 
        values = [input.value for input in inputs]
        return eval(strfunc)(*values)  
    def deserialize(self):
        return [ self.__dict__[key] for key in PARAMETER_INFO]

class Node (BaseClass):
    def __init__(self, *args, **kwargs):
          kwargs.update ({NODE_INFO[index] : value  for index, value in enumerate(args)})
          super(Node, self).__init__(*args, **kwargs)
          self.parameters  = {value.name : key for key, value in GLOBAL_PARAMETERS.items() if (value.nodeId == self.id)}
    def process(self):
        map (lambda (key, value): GLOBAL_PARAMETERS[value].eval(), self.parameters.items())
    def deserialize(self):
        return [ self.__dict__[key] for key in NODE_INFO]


# The idea is to create some examples and store the instances into nodes to save them and load the later again 

########################
## LINKED NODES GRAPH ##
########################

# Now the tool or the graphic tool will use these templates in order to create instances of them.
# Also these templates could be generated by using assemblies.

templatesDB = {
    "nodes": [ ["1", "sum"], 
               ["2", "substract"] ],
    "Parameters" : [ ["11", "1", "x", "float", 0, "public", None, 0 ],
                     ["12", "1", "y", "float", 0, "public", None, 0 ],
                     ["13", "1", "output", "function", 0, "public",  "lambda x, y: sum (x,y)", 0 ] ,
                     ["21", "2", "x", "float", 0, "public", None, 0 ],
                     ["22", "2", "y", "float", 0, "public", None, 0 ],
                     ["23", "2", "output", "function", 0, "public",  "lambda x, y: substract (x,y)", 0 ] ]
    }

# Instances are uses of the Templates already created
# The way the tool will create instances will be adding some extraid in order to differenciate one from another 

instancesDB = {
    "nodes": [ ["I1", "mysum"], 
               ["I2", "mysubstract"] ],
    "parameters" : [ ["I11", "I1", "x", "float", 0, "public", None, 30 ],
                     ["I12", "I1", "y", "float", 0, "public", None, 3.6 ],
                     ["I13", "I1", "output", "function", 0, "public",  "lambda x, y: sum (x,y)", 0 ] ,
                     ["I21", "I2", "x", "float", 0, "public", "I13", 0 ],
                     ["I22", "I2", "y", "float", 0, "public", None, 2.6 ],
                     ["I23", "I2", "output", "function", 0, "public",  "lambda x, y: substract (x,y)", 0 ] ]
    } # Result must be 31


# Create a dict with all parameters and nodes from the data source (DB, files, web reques, etc..)
GLOBAL_PARAMETERS = {param[0] : Parameter(*param) for param in instancesDB["parameters"]}
GLOBAL_NODES = {node[0] : Node(*node) for node in instancesDB["nodes"]}

#####################################
## Deserialzie the objects (again) ##
#####################################

#for key, value in GLOBAL_PARAMETERS.items():
#    print (value.deserialize())
#for key, value in GLOBAL_NODES.items():
#    print (value.deserialize())

# CREATE JSON with the current instances
#json_instances = { "nodes" : [value.deserialize() for key, value in GLOBAL_NODES.items() ],
#                   "parameters" :  [value.deserialize() for key, value in GLOBAL_PARAMETERS.items() ] }

#with open('data.txt', 'w') as outfile:
#    json.dump(json_instances, outfile)

##########################
## LOAD AGAIN FROM JSON ##
##########################

with open('data.txt', 'r') as infile:
    instancesDB = json.load(infile)

# Create a dict with all parameters and nodes from the data source (DB, files, web reques, etc..)
GLOBAL_PARAMETERS = {param[0] : Parameter(*param) for param in instancesDB["parameters"]}
GLOBAL_NODES = {node[0] : Node(*node) for node in instancesDB["nodes"]}

#######################
## Process the graph ##
#######################

#Nodes["I1"].process()
#Nodes["I2"].process()
#print (Parameters["I13"].value)
#print (Parameters["I23"].value)

GLOBAL_NODES["I2"].process()
print (GLOBAL_PARAMETERS["I13"].value)
print (GLOBAL_PARAMETERS["I23"].value)

print ("END")

