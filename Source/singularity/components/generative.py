from ..core import *
from .nodes import *
import copy




##########################
# Boltzmann Machine Node #
##########################

class RBMNode(Node):
    #constructor
    def __init__(self, dimension, 
                input_shapes = None, init_weights = glorot_uniform, init_bias = zeros , 
                inputs = None, activation = sigmoid, trainable = True, name = None):
        super(RBMNode,self).__init__(input_shapes, init_weights, init_bias, inputs, activation, trainable, shared, name)
        self._dimension =  dimension      
    ################################################
    #overridemethod
    def _get_shape_from_inputs(self):
        return (None, self._dimension)
    #overridemethod
    def _prepare_params(self, input_node):
        W = variable(self._init_weights((input_node.shape[1], self._dimension) ))
        b = variable(self._init_bias((self._dimension,)))
        c = variable(self._init_bias((self._dimension,)))
        return {'W': W, 'b':b, 'c':c}
    #overridemethod
    def _pre_activation(self, data, params):
        return dot(data, params['W']) + params['b']
    def _pre_backward_activation(self, data, params):
        return dot(data, params['W'].T) + params['c']