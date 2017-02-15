
class BaseNode(object):
    def __init__(self, shape = None, inputs = None, activation = None, name = None):
        self._inputs = inputs
        self._shape = shape
        self._activation = activation
        self._outputs = None
        self._name = name
    ################################################
    #abstractmethod
    def _prepare_outputs(self, data, index):
       return data
    #abstractmethod
    def _post_activation(self, outputs):
        return outputs
    #abstractmethod
    def initialize(self):
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


class InputNode (BaseNode):
    def __init__(self, shape):
        super(InputNode,self).__init__(shape)
        self._outputs = None




        
class BaseNode(object):
    def __init__(self, shape = None, inputs = None, activation = None, name = None):
        self._inputs = inputs
        self._shape = shape
        self._activation = activation
        self._outputs = None
        self._name = name
    ################################################
    #abstractmethod
    def _prepare_outputs(self, data, index):
       return data
    #abstractmethod
    def _post_activation(self, outputs):
        return outputs
    #abstractmethod
    def initialize(self):
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


class InputNode (BaseNode):
    def __init__(self, shape = None):
        super(InputNode,self).__init__(shape)
        self._outputs = None
    def get_something(self):
        return "Hoola"

f = BaseNode (shape = (None, 4))
print f.shape

input = InputNode()
print input.get_something()
print input.shape

f.__class__ = InputNode

print f.get_something()
print f.shape
