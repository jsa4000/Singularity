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

configuration = {
    "nodes": [
            {"class": "node", 
            "name": "sum",
            "x":  { "class": "parameter", 
                    "name": "x", 
                    "dtype": "float", 
                    "slot": 0,  
                    "category": "public", 
                    "origin": None, 
                    "value": 0
                    },
            "y": {  "class": "parameter", 
                    "name": "y", 
                    "dtype": "float", 
                    "slot": 0,  
                    "category": "public", 
                    "origin": None, 
                    "value": 0
                    },
            "output": { "class": "parameter", 
                        "name": "output", 
                        "dtype": "function", 
                        "slot": 1,  
                        "category": "public", 
                        "origin": "lambda x, y: sum (x,y)", 
                        "value": 0
                        }
            },
            {"class": "node", 
            "name": "substract",
            "x":  { "class": "parameter", 
                    "name": "x", 
                    "dtype": "float", 
                    "slot": 0,  
                    "category": "public", 
                    "origin": None, 
                    "value": 0
                    },
            "y": {  "class": "parameter", 
                    "name": "y", 
                    "dtype": "float", 
                    "slot": 0,  
                    "category": "public", 
                    "origin": None, 
                    "value": 0
                    },
            "output": { "class": "parameter", 
                        "name": "output", 
                        "dtype": "function", 
                        "slot": 1,  
                        "category": "public", 
                        "origin": "lambda x, y: substract (x,y)", 
                        "value": 0
                        }
            }
    ]
}

# Now the tool or the graphic tool will use these templates in order to create instances of them.
# Also these templates could be generated by using assemblies.
 
# The way the tool will create instances will be adding some extraid in order to differenciate one from another 

def sum (x,y):
    return x + y

def substract (x,y):
    return x - y

# This instance will be used to create the instances
def generateUId():
    return uuid.uuid4().get_hex()

class BaseClass (object):
    def __init__(self, *args, **kwargs):
        if ("id" not in kwargs):
            kwargs["id"] = generateUId()
        self.__dict__.update(kwargs)
    def deserialize(self):
        return {key: str(value) for (key, value) in self.__dict__.items() }
  
class parameter (BaseClass):
    def __init__(self, *args, **kwargs):
        if (len (args) > 0 and "nodeId" not in self.__dict__):
            self.__dict__["nodeId"] = args[0]
        super(parameter, self).__init__(*args, **kwargs)
    def eval(self):
        if (isinstance(self.origin, parameter)):
            self.origin.eval()
            self.value = self.origin.value
        elif (self.dtype == "function"):
            self.value = self._evaluate_function(self.origin)
    def _evaluate_function (self,strfunc):
        params = strfunc.split(":")[0].replace ("lambda","", 1).strip().split(",")
        inputs = [Nodes[self.nodeId].get(param.strip()) for param in params]
        map(lambda input: input.eval(), inputs) 
        values = [input.value for input in inputs]
        return eval(strfunc)(*values)  
    def deserialize(self):
        result = {}
        for (key, value) in self.__dict__.items():
            if (isinstance(value,parameter)):
                result[key] = value.id()
            else:
                result[key] = value
        return result  

class BaseNode (BaseClass):
    def __init__(self, *args, **kwargs):
        super(BaseNode, self).__init__(*args, **kwargs)
        for (key, value) in self.__dict__.items():
            if (isinstance(value, dict) and value["class"] == parameter.__name__):
                self.__dict__[key] = parameter(self.id, **value)
    def get (self, name):
        return self.__dict__[name]
    def process(self):
        for (key, value) in self.__dict__.items():
            if (isinstance(value,parameter)):
                value.eval()
    def deserialize(self):
        result = {}
        for (key, value) in self.__dict__.items():
            if (isinstance(value,parameter)):
                result[key] = value.deserialize()
            else:
               result[key] = value
        return result

# The idea is to create some examples and store the instances into nodes to save them and load the later again 

#Lsit with all the nodes on the graph
Nodes = {}

tempSumNode = configuration["nodes"][0]

##########################################
## Create a Node and store it in a file ##
##########################################

#node = BaseNode (**tempSumNode)
#data = node.deserialize()
#print (json.dumps(data, indent=4))
#with open('data.txt', 'w') as outfile:
#    json.dump(data, outfile)

## Add this into the list of nodes
#Nodes[node.id] = node

########################################
## Now loads the instance from a file ##
########################################

#with open('data.txt', 'r') as infile:
#    tempSumNode = json.load(infile)

#node = BaseNode (**tempSumNode)
#Nodes[node.id] = node

#######################
## Process the graph ##
#######################

node.process()

print ("END")

