from ..core import *

#############
# Base Node #
#############  

class BaseNode(BaseClass):
    #constructor
    def __init__(self, shape= None, inputs = None, activation = None, trainable = False, shared = False, name = None, **kwargs):
        params = extract_args_kwargs(locals())
        params["inputs"] = self._get_nodes(params["inputs"]) 
        params["outputs"] = None
        params["params"] = []
        super(BaseNode, self).__init__(**params)
    #constructor (callable)
    def __call__(self, inputs = None, trainable = None, activation = None, shared = None):
        self._initialize(inputs, activation, trainable, shared)
        return self
    ################################################
    #abstractmethod
    def _pre_activation(self, data, params):
       return data
    #abstractmethod
    def _pre_back_activation(self, data, params):
       return data
    #abstractmethod
    def _post_activation(self, outputs):
        return outputs
    #abstractmethod
    def initialize(self):
        pass
    #abstractmethod
    def _prepare_params(self, input_node):
        return None
    #abstractmethod
    def _get_shape_from_inputs(self):
        return None
    #abstractmethod
    def _set_params(self, params):
        pass
    ################################################
    @property
    def name(self):
        return self._name   
    @property
    def shape(self):
        return self._shape
    @property
    def inputs(self):
        return self._inputs 
    @property
    def outputs(self):
        return self._outputs
    @property
    def params(self):
        return self._get_params()
    @property
    def trainable(self):
        return self._trainable    
    ################################################
    #publicmethod
    def set_params(self, params):
        self._set_params(params)
    #publicmethod
    def forward(self, data):
        return self._forward(data)
    #publicmethod
    def backward(self, data):
        return self._backward(data)
    ################################################
    def _set_settings(self, settings): 
       pass
    def _get_settings(self, settings): 
       return None
    #privatemethod
    def _initialize(self, inputs, activation, trainable, shared):
        if (inputs is not None):
            self._inputs = self._get_nodes(inputs)
            self._shape = self._get_shape_from_inputs()
        if (self._shape is None ):
            self._shape = self._get_shape_from_inputs()
        if (activation is not None):
            self._activation = activation
        if (trainable is not None):
            self._trainable = trainable
        if (shared is not None):
            self._shared = shared
        if (self._trainable and is_none_or_empty(self._params)):
            self._initialize_params()
    #privatemethod
    def _forward(self, data):
        if (not is_none_or_empty(self._params)):
            if (is_collection(data)):
                result = []
                for index, item in enumerate(data):
                    if (self._shared):
                        result.append(self._pre_activation(item, self._params[0]))
                    else:
                        result.append(self._pre_activation(item, self._params[index]))
            else:
                result = self._pre_activation(data, self._params[0])
        else:
            result = data
        if (self._activation is not None):
            result = self._activation(result)
        self._outputs = self._post_activation(result)
        return self._outputs
    #privatemethod
    def _backward(self, data):
        if (not is_none_or_empty(self._params)):
            if (is_collection(data)):
                result = []
                for index, item in enumerate(data):
                    if (self._shared):
                        result.append(self._pre_back_activation(item, self._params[0]))
                    else:
                        result.append(self._pre_back_activation(item, self._params[index]))
            else:
                result = self._pre_back_activation(data, self._params[0])
        else:
            result = data
        if (self._activation is not None):
            result = self._activation(result)
        self._outputs = self._post_activation(result)
        return self._outputs
    #privatemethod
    def _initialize_params(self):
        self._params = []
        if (self._shared):
            self._params.append(self._prepare_params(self._inputs[0]))
        else:
            for input_node in self._inputs:
                self._params.append(self._prepare_params(input_node))
    #privatemethod
    def _get_params(self):
        if (is_none_or_empty(self._params)):
            result = []
        if (self._shared):
            result = self._params[0].values()
        else:
            result = []
            for param in self._params:
                result.extend (param.values())
        return result
    #privatemethod
    def _get_nodes(self, nodes):
        if (is_none_or_empty(nodes)):
            result = []
        elif (isinstance(nodes, BaseNode)):
            result = []
            result.append(nodes)
        else:
            result = nodes
        return result

##############
# Input Node #
##############

class InputNode (BaseNode):
    #constructor
    def __init__(self, shape = None):
        params = extract_args_kwargs(locals())
        super(InputNode, self).__init__(**params)
        self._outputs = placeholder(self._shape)
                    
########
# Node #
########

class Node(BaseNode):
    #constructor
    def __init__(self, input_shapes = None, init_weights = None, init_bias = None, **kwargs):
        params = extract_args_kwargs(locals())
        super(Node, self).__init__(**params)
        if (not is_none_or_empty(input_shapes)):
            self._inputs.extend(self._get_inputs_from_shapes(input_shapes))
    ################################################
    #privatemethod
    def _get_inputs_from_shapes(self, input_shapes):
        if (input_shapes == None):
            inputs_shapes = []
        elif (isinstance(input_shapes, tuple)):
            inputs_shapes = []
            inputs_shapes.append(input_shapes)
        else:
            inputs_shapes = input_shapes
        result = []
        for input_shape in inputs_shapes:
            result.append(InputNode(input_shape))
        return result


