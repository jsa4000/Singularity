from ..core import *
from ..utils.hdf5 import *
from .layers import *

#######@######
# Graph Model #
########@##### 

class GraphModel (object):
    def __init__(self, inputs = None, outputs  = None, loss = None, optimizer = None, regularizers = [], name = None):
        self._inputs = self._get_nodes(inputs)
        self._outputs = outputs
        self._loss = loss
        self._optimizer = optimizer
        self._regularizers = regularizers
        self._name = name
        self._compiled = False
        self._predict = None
        self._train = None
    ################################################          
    def _forward_propagation (self, x, node):
        if (isinstance(node, InputNode)):
            y = node.outputs
            if (node not in self._inputs ):
                self._inputs.append(node)
        elif (len(node.inputs) > 1):
            y = []
            for input in node.inputs:
                 y.append(self._forward_propagation(x, input))
        else:
            y = self._forward_propagation(x, node.inputs[0])
        return node.forward(y)
    def _build_nodes (self, node):
        for input in node.inputs:
            self._build_nodes(input)
        node()
    def _get_inputs (self, node, inputs):
        if (isinstance(node, InputNode)):
            inputs.append(node)
        else:
            for input in node.inputs:
                self._get_inputs(input, inputs)
        return inputs
    def _get_all_params (self, node, params):
        if (not (isinstance(node, InputNode))):
            if (node.trainable):
                params.extend(node.params)
            for input in node.inputs:
                self._get_all_params(input, params)
        return params
    def _get_nodes_configuration (self, node, config):
        config[node.name] = NodeConfig(node.name, type(node).__name__,node.params)
        for input in node.inputs:
            self._get_nodes_configuration(input, config)
        return config
    def _set_nodes_configuration (self, node, config):
        if (node.name in config):
            node.set_params(config[node.name].params)
        for input in node.inputs:
            self._set_nodes_configuration(input, config)
    def _get_regularization (self, regularizers):
        return sum(list(regularizer(self.get_all_params()) for regularizer in regularizers ))
    def _build(self, loss, optimizer, regularizers, build_nodes = True):
        if (loss is not None):
            self._loss = loss
        if (optimizer is not None):
            self._optimizer = optimizer
        if (regularizers is not None):
            self._regularizers = regularizers
        if (build_nodes):        
            self._build_nodes (self._outputs)
        predict = symbolicfunction(self._forward_propagation)(None, self._outputs)
        if (self._inputs is None):
            self._inputs = self._get_inputs(self._outputs, [])        
        inputs = list(input.outputs for input in self._inputs)
        self._predict = function(inputs,predict)
        if (self._loss is not None):
            t = placeholder(self._outputs.shape)
            cost = symbolicfunction(loss)(predict,t) + self._get_regularization(regularizers)
            updates = optimizer.optimize(cost = cost, params = self.get_all_params())
            inputs.append(t)
            self._train = function(inputs, cost, updates = updates)
        self._compiled = True
    ################################################ 
    @property
    def name(self):
        return self._name   
    def load(self, filename):
        model = load_model(filename)
        self._set_nodes_configuration (self._outputs, model.nodes)
    def save(self, filename, compressed = True):
        save_model(filename,ModelConfig(self._name, self._get_nodes_configuration(self._outputs, {})), compressed)
    def build(self, loss = None, optimizer = None, regularizers = []):
        self._build(loss, optimizer, regularizers)
    def get_all_params (self):
         return self._get_all_params(self._outputs, [])
    def predict (self, inputs):
        if (self._compiled):
            return self._predict(inputs)
        else:
            raise singularity_exception("The model has not been compiled yet")   
    def train (self,x_train, t_train, batch_size= 50, iterations = 3):
        for i in range(iterations):
            print "iteration %d" % (i+ 1)
            for start in range(0, len(x_train), batch_size):
                x_batch= x_train[start:start+ batch_size]
                t_batch= t_train[start:start+ batch_size]
                cost = self._train(x_batch, t_batch)
                print "cost: %.5f" % cost  
    def accuracy (self,x_test, y_test ):
        predictions_test= np.argmax(self._predict(x_test), axis=1)
        accuracy = np.mean(predictions_test== y_test)
        print "accuracy: %.5f" % accuracy
        return  accuracy
    ################################################
    def _get_nodes(self, nodes):
        if (nodes == None):
            result = []
        elif (isinstance(nodes, BaseNode)):
            result = []
            result.append(nodes)
        else:
            result = nodes
        return result 

#######@#########
# Overlay Model #
########@######## 

class OverlayModel (GraphModel):
    def __init__(self):
        super(OverlayModel,self).__init__()
        self._layers = []
    @property
    def layers(self):
        return self._layers
    def build(self, loss = None, optimizer = None, regularizers = []):
        if (isinstance (self._layers[0], InputLayer)):
            self._inputs = self._get_nodes(self._layers[0])
        else:
            self._layers[0]()
        for previouslayer, currentlayer in zip(self._layers,self._layers[1:]):
            currentlayer(previouslayer)
        self._outputs = self._layers[-1]
        self._build(loss, optimizer, regularizers,False)
    def count(self):
         return len(self._layers)
    def add(self, layer):
        if (isinstance(layer,BaseNode)):
            self._layers.append(layer)
        else:
            raise singularity_exception("Elements added must inherit from the base Layer clas.")  
    def insert(self, i, layer):
        if (isinstance(layer,BaseNode)):
            self._layers.insert(i,InputLayer)
        else:
            raise singularity_exception("Elements added must inherit from the base Layer class.")   
    def removeAt (self, i):
        if (len(self._layers) > i):
            layer = self._layers[i]
            self._layers.remove(layer)
        else:
            raise singularity_exception("Index higher than the elements in the model.")   
    def remove (self, layer):
        if (layer in self._layers):
            self._layers.remove(layer)
        else:
            raise singularity_exception("Layer not founded in the model.")   
    def extend(self, layers):
        self._layers.extend(layers)       
